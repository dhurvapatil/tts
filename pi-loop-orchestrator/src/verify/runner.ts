/**
 * Verification runner.
 *
 * The verification gate is the only objective ground truth in the loop.
 * An LLM judge that "looks done" is not enough — we shell out to the real
 * build/test command and trust the exit code. Only after tests pass do we
 * optionally consult the reviewer LLM for quality/blocking issues.
 */

import { spawn } from "node:child_process";
import type { VerificationResult } from "../loop/state.ts";

const MAX_OUTPUT = 30 * 1024; // 30KB cap per stream (test output can be huge)

export interface RunVerificationOptions {
	command: string;
	cwd: string;
	timeoutMs: number;
	signal?: AbortSignal;
}

export function runVerification(opts: RunVerificationOptions): Promise<VerificationResult> {
	return new Promise((resolve) => {
		const startedAt = Date.now();
		let stdout = "";
		let stderr = "";
		let timedOut = false;
		let killed = false;

		const proc = spawn(opts.command, {
			cwd: opts.cwd,
			shell: true, // need a shell so strings like "npm test" work
			stdio: ["ignore", "pipe", "pipe"],
			windowsHide: true,
		});

		proc.stdout.on("data", (data) => {
			stdout = appendCapped(stdout, data.toString(), MAX_OUTPUT);
		});
		proc.stderr.on("data", (data) => {
			stderr = appendCapped(stderr, data.toString(), MAX_OUTPUT);
		});

		const timer = setTimeout(() => {
			timedOut = true;
			proc.kill("SIGTERM");
			setTimeout(() => {
				if (!proc.killed) proc.kill("SIGKILL");
			}, 5000);
		}, opts.timeoutMs);

		if (opts.signal) {
			const abort = () => {
				killed = true;
				proc.kill("SIGTERM");
				setTimeout(() => {
					if (!proc.killed) proc.kill("SIGKILL");
				}, 5000);
			};
			if (opts.signal.aborted) abort();
			else opts.signal.addEventListener("abort", abort, { once: true });
		}

		proc.on("close", (code) => {
			clearTimeout(timer);
			const durationMs = Date.now() - startedAt;
			resolve({
				passed: code === 0 && !timedOut && !killed,
				command: opts.command,
				exitCode: code,
				stdout,
				stderr,
				durationMs,
				timedOut,
				killed,
			});
		});

		proc.on("error", (err) => {
			clearTimeout(timer);
			resolve({
				passed: false,
				command: opts.command,
				exitCode: null,
				stdout,
				stderr: (stderr ? stderr + "\n" : "") + err.message,
				durationMs: Date.now() - startedAt,
				timedOut: false,
				killed: false,
			});
		});
	});
}

function appendCapped(current: string, chunk: string, maxBytes: number): string {
	const combined = current + chunk;
	if (Buffer.byteLength(combined, "utf8") <= maxBytes) return combined;
	// Cap and mark truncation
	const truncatedNote = `\n[...output truncated to ${formatSize(maxBytes)}]`;
	let out = combined;
	while (Buffer.byteLength(out, "utf8") + Buffer.byteLength(truncatedNote, "utf8") > maxBytes) {
		out = out.slice(0, -1);
	}
	return out + truncatedNote;
}

function formatSize(bytes: number): string {
	if (bytes < 1024) return `${bytes}B`;
	if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
	return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
}
