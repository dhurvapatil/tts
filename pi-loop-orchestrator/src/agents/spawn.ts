/**
 * Agent spawn layer.
 *
 * Forks the spawn pattern from pi's subagent extension: each agent invocation
 * is a separate `pi` subprocess running in `--mode json` with an appended
 * system prompt from the bundled .md files. This gives each variant an
 * isolated context window and structured output we can parse.
 *
 * The fan-out executor walks the blueprint's FanOutStep[] sequentially.
 * Within a single step, if `count > 1`, the variants run in parallel.
 */

import { spawn } from "node:child_process";
import * as fsp from "node:fs/promises";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { AgentReport, FanOutStep } from "../loop/state.ts";
import { loadAgentDef } from "./roles.ts";

interface RunOneOptions {
	step: FanOutStep;
	variant: number;
	stateContext: StateContext;
	cwd: string;
	signal: AbortSignal | undefined;
}

export interface StateContext {
	attempt: number;
	focus?: string;
	objective: string;
	lastReports?: string;
	lastVerification?: string;
	verifyCommand: string;
}

const MAX_OUTPUT_PER_REPORT = 30 * 1024;

function getPiInvocation(): { command: string; args: string[] } {
	const currentScript = process.argv[1];
	const isBunVirtualScript = currentScript?.startsWith("/$bunfs/root/");
	if (currentScript && !isBunVirtualScript && fs.existsSync(currentScript)) {
		return { command: process.execPath, args: [currentScript] };
	}
	const execName = path.basename(process.execPath).toLowerCase();
	const isGenericRuntime = /^(node|bun)(\.exe)?$/.test(execName);
	if (!isGenericRuntime) {
		return { command: process.execPath, args: [] };
	}
	return { command: "pi", args: [] };
}

async function runOne(opts: RunOneOptions): Promise<AgentReport> {
	const { step, variant, stateContext, cwd, signal } = opts;
	const def = await loadAgentDef(step.role);
	const systemPrompt = def?.systemPrompt ?? defaultFallbackPrompt(step.role);
	const model = step.model ?? def?.model;
	const tools = step.tools ?? def?.tools;
	const task = renderTask(step.task, {
		...stateContext,
		variant,
		count: step.count ?? 1,
	});

	const tmpDir = await fsp.mkdtemp(path.join(os.tmpdir(), "pi-loop-"));
	const promptPath = path.join(tmpDir, "system-prompt.md");
	await fsp.writeFile(promptPath, systemPrompt, { encoding: "utf8", mode: 0o600 });

	const args: string[] = [
		"--mode",
		"json",
		"-p",
		"--no-session",
		"--append-system-prompt",
		promptPath,
		// Prevent the child pi from re-loading our loop orchestrator and
		// recursing. The child has no business calling loop_* tools; those
		// are for the orchestrator LLM only.
		"--exclude-tools",
		"loop_design,loop_start,loop_run_iteration,loop_get_state,loop_stop",
	];
	if (model) args.push("--model", model);
	if (tools && tools.length > 0) args.push("--tools", tools.join(","));
	args.push(`Task: ${task}`);

	const invocation = getPiInvocation();

	const startedAt = Date.now();
	let finalText = "";
	let exitCode = 0;
	let stopReason: string | undefined;
	let errorMessage: string | undefined;
	let modelSeen: string | undefined;
	let stderrBuf = "";
	const usage = { input: 0, output: 0, cost: 0, turns: 0 };

	const result: AgentReport = await new Promise((resolve) => {
		const proc = spawn(invocation.command, [...invocation.args, ...args], {
			cwd,
			shell: false,
			stdio: ["ignore", "pipe", "pipe"],
			windowsHide: true,
		});

		let buffer = "";
		proc.stdout.on("data", (data) => {
			buffer += data.toString();
			const lines = buffer.split("\n");
			buffer = lines.pop() ?? "";
			for (const line of lines) {
				if (!line.trim()) continue;
				let event: any;
				try {
					event = JSON.parse(line);
				} catch {
					continue;
				}
				if (event.type === "message_end" && event.message) {
					const msg = event.message;
					if (msg.role === "assistant") {
						usage.turns++;
						if (msg.usage) {
							usage.input += msg.usage.input || 0;
							usage.output += msg.usage.output || 0;
							usage.cost += msg.usage.cost?.total || 0;
						}
						if (msg.model && !modelSeen) modelSeen = msg.model;
						if (msg.stopReason) stopReason = msg.stopReason;
						if (msg.errorMessage) errorMessage = msg.errorMessage;
						for (const part of msg.content ?? []) {
							if (part.type === "text") finalText = part.text;
						}
					}
				}
			}
		});

		proc.stderr.on("data", (data) => {
			stderrBuf += data.toString();
		});

		proc.on("close", (code) => {
			exitCode = code ?? 1;
			resolve({
				role: step.role,
				variant,
				model: modelSeen,
				output: truncate(finalText || stderrBuf, MAX_OUTPUT_PER_REPORT),
				exitCode,
				stopReason,
				errorMessage,
				usage,
				startedAt,
				finishedAt: Date.now(),
			});
		});

		proc.on("error", (err) => {
			resolve({
				role: step.role,
				variant,
				output: err.message,
				exitCode: 1,
				errorMessage: err.message,
				startedAt,
				finishedAt: Date.now(),
			});
		});

		if (signal) {
			const kill = () => {
				proc.kill("SIGTERM");
				setTimeout(() => {
					if (!proc.killed) proc.kill("SIGKILL");
				}, 5000);
			};
			if (signal.aborted) kill();
			else signal.addEventListener("abort", kill, { once: true });
		}
	});

	// Cleanup
	try {
		fs.unlinkSync(promptPath);
		fs.rmdirSync(tmpDir);
	} catch {
		// best effort
	}

	return result;
}

/**
 * Run the full fan-out: each step in sequence, with `count` parallel
 * variants within a step.
 */
export async function runFanOut(
	fanOut: FanOutStep[],
	stateContext: StateContext,
	cwd: string,
	signal: AbortSignal | undefined,
): Promise<AgentReport[]> {
	const all: AgentReport[] = [];
	for (let i = 0; i < fanOut.length; i++) {
		const step = fanOut[i];
		const count = step.count ?? 1;
		const expanded = Array.from({ length: count }, (_, v) => ({ step, variant: v }));
		const reports = await Promise.all(expanded.map((e) => runOne({ ...e, stateContext, cwd, signal })));
		all.push(...reports);
	}
	return all;
}

/**
 * Render a task template. Supports:
 *   {{key}}              - simple value substitution
 *   {{#if key}}...{{/if}} - conditional block (omitted if value is empty/null)
 *
 * The conditional syntax is intentionally minimal — we want blueprint
 * authors to be able to write readable templates without pulling in a
 * Mustache library.
 */
export function renderTask(template: string, ctx: Record<string, any>): string {
	const condRe = /\{\{#if (\w+)\}\}([\s\S]*?)\{\{\/if\}\}/g;
	const rendered = template.replace(condRe, (_, key, body) => {
		const v = ctx[key];
		if (v === undefined || v === null || v === "") return "";
		return body;
	});
	return rendered.replace(/\{\{(\w+)\}\}/g, (_, key) => {
		const v = ctx[key];
		if (v === undefined || v === null) return "";
		return String(v);
	});
}

function defaultFallbackPrompt(role: string): string {
	return [
		`You are the **${role}** in a build/test/fix loop.`,
		"Read files, make targeted changes (if your role requires it), and report what you did.",
		"Be concise. End with a structured summary of your actions and findings.",
	].join("\n");
}

function truncate(s: string, maxBytes: number): string {
	if (Buffer.byteLength(s, "utf8") <= maxBytes) return s;
	let out = s;
	while (Buffer.byteLength(out, "utf8") > maxBytes) {
		out = out.slice(0, -1);
	}
	return out + "\n[...truncated]";
}
