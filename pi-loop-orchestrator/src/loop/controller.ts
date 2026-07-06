/**
 * Loop controller. This is the orchestrator's own logic — it's what the
 * root agent invokes (indirectly, via the loop_run_iteration tool) to
 * drive a build/test/fix loop forward.
 *
 * Each call to runIteration:
 *   1. Loads state, applies stop rules (max iterations, max duration).
 *   2. Builds this iteration's prompts from the blueprint's task templates,
 *      parameterized by the previous iteration's reports + verification.
 *   3. Runs the fan-out (sequential steps, parallel variants within a step).
 *   4. Runs the verification gate (real shell command, exit code matters).
 *   5. Computes a recommendation (continue / stop / fail).
 *   6. Persists the iteration record, updates state, returns the result.
 *
 * No-progress detection: we hash (exit code, top failing test names) from
 * the last N verification results. If they match, increment noProgressCount;
 * if the count exceeds stopRules.noProgressLimit, stop with reason.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	type LoopBlueprint,
	type LoopState,
	type IterationRecord,
	type AgentReport,
	type VerificationResult,
	type Recommendation,
	loadLoopState,
	saveLoopState,
	newLoopState,
	listLoopStates,
} from "./state.ts";
import { runFanOut, type StateContext } from "../agents/spawn.ts";
import { runVerification } from "../verify/runner.ts";

export async function startLoop(blueprint: LoopBlueprint, _cwd: string, _pi: ExtensionAPI): Promise<string> {
	const state = newLoopState(blueprint);
	await saveLoopState(state);
	return state.id;
}

export async function getLoopState(id: string): Promise<LoopState> {
	return loadLoopState(id);
}

export async function listLoops() {
	return listLoopStates();
}

export async function stopLoop(id: string, reason: string): Promise<LoopState> {
	const state = await loadLoopState(id);
	state.status = "stopped";
	state.stopReason = reason;
	await saveLoopState(state);
	return state;
}

export interface IterationResult {
	loopId: string;
	attempt: number;
	passed: boolean;
	verification: VerificationResult;
	recommendation: Recommendation;
	agentReports: AgentReport[];
	status: LoopState["status"];
	durationMs: number;
}

/**
 * Run a single iteration of the loop. The orchestrator agent is expected
 * to call this repeatedly until the returned `recommendation.action` is
 * "stop" (or "fail" if it can't recover).
 */
export async function runIteration(
	loopId: string,
	focus: string | undefined,
	cwd: string,
	_pi: ExtensionAPI,
	signal: AbortSignal | undefined,
): Promise<IterationResult> {
	const state = await loadLoopState(loopId);
	const startTime = Date.now();

	// If the loop is already terminal, just return the current state.
	if (state.status !== "running") {
		const last = state.history[state.history.length - 1];
		return {
			loopId,
			attempt: state.history.length,
			passed: state.status === "completed",
			verification: last?.verification ?? emptyVerification("loop is " + state.status),
			recommendation: {
				action: "stop",
				reason: `loop is ${state.status}${state.stopReason ? ": " + state.stopReason : ""}`,
			},
			agentReports: last?.agentReports ?? [],
			status: state.status,
			durationMs: 0,
		};
	}

	// Stop rule: max iterations
	if (state.history.length >= state.blueprint.stopRules.maxIterations) {
		state.status = "failed";
		state.stopReason = `max iterations reached (${state.history.length})`;
		await saveLoopState(state);
		return {
			loopId,
			attempt: state.history.length,
			passed: false,
			verification: emptyVerification(state.stopReason),
			recommendation: { action: "stop", reason: state.stopReason },
			agentReports: [],
			status: state.status,
			durationMs: 0,
		};
	}

	// Stop rule: max duration
	const elapsed = Date.now() - state.startedAt;
	if (elapsed >= state.blueprint.stopRules.maxDurationMs) {
		state.status = "failed";
		state.stopReason = `max duration reached (${Math.round(elapsed / 1000)}s of ${Math.round(state.blueprint.stopRules.maxDurationMs / 1000)}s)`;
		await saveLoopState(state);
		return {
			loopId,
			attempt: state.history.length,
			passed: false,
			verification: emptyVerification(state.stopReason),
			recommendation: { action: "stop", reason: state.stopReason },
			agentReports: [],
			status: state.status,
			durationMs: 0,
		};
	}

	const attempt = state.history.length + 1;
	const lastIteration = state.history[state.history.length - 1];

	// Build state context for prompt templates
	const stateContext: StateContext = {
		attempt,
		focus: focus || lastIteration?.recommendation.suggestedFocus,
		objective: state.blueprint.objective,
		lastReports: lastIteration ? summarizeReports(lastIteration.agentReports) : "",
		lastVerification: lastIteration ? summarizeVerification(lastIteration.verification) : "",
		verifyCommand: state.blueprint.stopRules.verifyCommand,
	};

	// Step 1: Fan-out
	const agentReports = await runFanOut(state.blueprint.fanOut, stateContext, cwd, signal);

	// If all agents crashed before producing output, fail fast.
	const anyUsableReport = agentReports.some((r) => r.output && r.output.trim().length > 0);
	if (!anyUsableReport && agentReports.length > 0) {
		state.status = "failed";
		state.stopReason = "all agents produced no usable output";
		await saveLoopState(state);
		return {
			loopId,
			attempt,
			passed: false,
			verification: emptyVerification(state.stopReason),
			recommendation: {
				action: "fail",
				reason: state.stopReason + " — check agent configuration and try again",
			},
			agentReports,
			status: state.status,
			durationMs: Date.now() - startTime,
		};
	}

	// Step 2: Verification (the objective ground truth)
	const verification = await runVerification({
		command: state.blueprint.stopRules.verifyCommand,
		cwd: state.blueprint.stopRules.verifyCwd ?? cwd,
		timeoutMs: state.blueprint.stopRules.verifyTimeoutMs,
		signal,
	});

	// Step 3: No-progress detection
	const signature = failureSignature(verification);
	if (signature && signature === state.lastFailureSignature) {
		state.noProgressCount++;
	} else {
		state.noProgressCount = signature ? 0 : state.noProgressCount;
	}
	state.lastFailureSignature = signature ?? state.lastFailureSignature;

	// Step 4: Compute recommendation
	const recommendation = computeRecommendation(state, verification, focus, attempt);

	// Persist
	const record: IterationRecord = {
		attempt,
		focus: stateContext.focus,
		startedAt: startTime,
		finishedAt: Date.now(),
		fanOutSpec: state.blueprint.fanOut,
		agentReports,
		verification,
		recommendation,
	};
	state.history.push(record);

	if (recommendation.action === "stop") {
		state.status = verification.passed ? "completed" : "failed";
		state.stopReason = recommendation.reason;
	}

	await saveLoopState(state);

	return {
		loopId,
		attempt,
		passed: verification.passed,
		verification,
		recommendation,
		agentReports,
		status: state.status,
		durationMs: Date.now() - startTime,
	};
}

function computeRecommendation(
	state: LoopState,
	verification: VerificationResult,
	userFocus: string | undefined,
	attempt: number,
): Recommendation {
	// No-progress limit
	if (
		!verification.passed &&
		state.noProgressCount >= state.blueprint.stopRules.noProgressLimit
	) {
		return {
			action: "stop",
			reason: `no progress for ${state.noProgressCount} consecutive iterations — same failure pattern repeats`,
			suggestedFocus: "abandon this approach and try a fundamentally different strategy",
		};
	}

	// Max iterations (defensive: also checked at the top)
	if (attempt >= state.blueprint.stopRules.maxIterations) {
		return { action: "stop", reason: "max iterations reached" };
	}

	// Verification passed -> stop (success)
	if (verification.passed) {
		return { action: "stop", reason: "verification passed" };
	}

	// Verification failed -> continue with a focused retry
	const failingTests = extractFailingTests(verification);
	const focusText = userFocus
		? `user-supplied focus: ${userFocus}`
		: failingTests.length > 0
			? `address these failing tests: ${failingTests.slice(0, 5).join("; ")}`
			: `verification command exited ${verification.exitCode} — diagnose from stderr: ${(verification.stderr || verification.stdout).split("\n").slice(0, 3).join(" | ").slice(0, 300)}`;

	return {
		action: "continue",
		reason: `verification failed: ${failingTests.slice(0, 3).join("; ") || `exit ${verification.exitCode}` || "see stderr"}`,
		suggestedFocus: focusText,
	};
}

function failureSignature(v: VerificationResult): string | null {
	if (v.passed || v.timedOut) return null;
	const failing = extractFailingTests(v).slice(0, 3);
	return `${v.exitCode ?? "null"}|${failing.join("|")}`;
}

function extractFailingTests(v: VerificationResult): string[] {
	const out = v.stdout + "\n" + v.stderr;
	const failures: string[] = [];

	// Common test framework failure markers
	const patterns: RegExp[] = [
		/^FAIL\s+(\S+)/gm, // jest, mocha (line-anchored)
		/✕\s+(.+)/g, // jest spec failure
		/^FAILED\s+(\S+)/gm, // pytest, cargo
		/^Error:\s+(.+)/gm, // generic
		/^\s*(\d+)\s+(?:failed|FAIL)/gm, // summary lines
		/●\s+(.+)/g, // jest reporter
	];
	for (const p of patterns) {
		let m: RegExpExecArray | null;
		while ((m = p.exec(out)) !== null) {
			const s = m[1].trim().replace(/\s+/g, " ");
			if (s.length > 0 && s.length < 200) failures.push(s);
		}
	}
	return [...new Set(failures)].slice(0, 20);
}

function summarizeReports(reports: AgentReport[]): string {
	return reports
		.map((r) => {
			const variantTag = r.variant !== undefined && r.variant > 0 ? ` (variant ${r.variant})` : "";
			return `## [${r.role}${variantTag}] exit=${r.exitCode}${r.stopReason ? ` stopReason=${r.stopReason}` : ""}\n\n${r.output}`;
		})
		.join("\n\n---\n\n");
}

function summarizeVerification(v: VerificationResult): string {
	const tail = (s: string, n: number) => {
		const lines = s.split("\n");
		const last = lines.slice(-n).join("\n");
		return last.length > 2000 ? last.slice(0, 2000) + "\n[...truncated]" : last;
	};
	return [
		`exit=${v.exitCode} duration=${v.durationMs}ms timedOut=${v.timedOut} killed=${v.killed ?? false}`,
		`stdout (last lines):\n${tail(v.stdout, 20)}`,
		`stderr (last lines):\n${tail(v.stderr, 20)}`,
	].join("\n\n");
}

function emptyVerification(reason: string): VerificationResult {
	return {
		passed: false,
		command: "",
		exitCode: null,
		stdout: "",
		stderr: reason,
		durationMs: 0,
		timedOut: false,
	};
}
