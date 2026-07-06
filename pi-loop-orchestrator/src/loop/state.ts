/**
 * Loop state types and persistence.
 *
 * Each loop has a state object that persists between iterations to
 * `~/.pi/agent/loops/<id>.json`. The state survives pi restarts so a
 * long-running loop can be inspected or resumed.
 */

import * as fs from "node:fs";
import * as fsp from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

/**
 * One slot in the fan-out spec. Steps run sequentially; each step may
 * spawn `count` parallel variants of the same role.
 */
export interface FanOutStep {
	role: string;
	task: string; // template, supports {{attempt}}, {{focus}}, {{objective}}, {{lastReports}}, {{lastVerification}}, {{variant}}
	count?: number; // parallel variants (default 1)
	model?: string; // override agent default model
	tools?: string[]; // override agent default tools
	condition?: string; // reserved for future: "skip if ...". For now always runs.
}

export interface StopRules {
	maxIterations: number;
	maxDurationMs: number;
	noProgressLimit: number; // stop if last N iterations show no progress (same exit + similar stderr)
	verifyCommand: string;
	verifyTimeoutMs: number;
	verifyCwd?: string;
}

export interface LoopBlueprint {
	name?: string;
	description?: string;
	objective: string; // done-state, not activity
	fanOut: FanOutStep[];
	stopRules: StopRules;
}

export interface AgentReport {
	role: string;
	variant?: number;
	model?: string;
	output: string;
	exitCode: number;
	stopReason?: string;
	errorMessage?: string;
	usage?: { input: number; output: number; cost: number; turns: number };
	startedAt: number;
	finishedAt: number;
}

export interface VerificationResult {
	passed: boolean;
	command: string;
	exitCode: number | null;
	stdout: string;
	stderr: string;
	durationMs: number;
	timedOut: boolean;
	killed?: boolean;
	reviewer?: {
		approved: boolean;
		blockers: string[];
		notes: string;
		raw?: string;
	};
}

export interface Recommendation {
	action: "continue" | "stop" | "fail";
	reason: string;
	suggestedFocus?: string;
}

export interface IterationRecord {
	attempt: number;
	focus?: string;
	startedAt: number;
	finishedAt: number;
	fanOutSpec: FanOutStep[];
	agentReports: AgentReport[];
	verification: VerificationResult;
	recommendation: Recommendation;
}

export interface LoopState {
	id: string;
	blueprint: LoopBlueprint;
	status: "running" | "stopped" | "completed" | "failed";
	stopReason?: string;
	startedAt: number;
	updatedAt: number;
	history: IterationRecord[];
	// tracking for no-progress detection
	noProgressCount: number;
	lastFailureSignature?: string;
}

function loopsDir(): string {
	return path.join(os.homedir(), ".pi", "agent", "loops");
}

function ensureDirSync(): void {
	const dir = loopsDir();
	if (!fs.existsSync(dir)) {
		fs.mkdirSync(dir, { recursive: true });
	}
}

function loopPath(id: string): string {
	return path.join(loopsDir(), `${id}.json`);
}

export async function saveLoopState(state: LoopState): Promise<void> {
	ensureDirSync();
	state.updatedAt = Date.now();
	const target = loopPath(state.id);
	const tmp = `${target}.tmp`;
	await fsp.writeFile(tmp, JSON.stringify(state, null, 2), "utf8");
	await fsp.rename(tmp, target);
}

export async function loadLoopState(id: string): Promise<LoopState> {
	const content = await fsp.readFile(loopPath(id), "utf8");
	const state = JSON.parse(content) as LoopState;
	// Validate minimal shape
	if (!state.id || !state.blueprint || !Array.isArray(state.history)) {
		throw new Error(`Loop state ${id} is corrupt`);
	}
	return state;
}

export async function listLoopStates(): Promise<
	Array<{ id: string; status: string; objective: string; attempts: number; updatedAt: number }>
> {
	ensureDirSync();
	const dir = loopsDir();
	const files = await fsp.readdir(dir);
	const results: Array<{ id: string; status: string; objective: string; attempts: number; updatedAt: number }> = [];
	for (const file of files) {
		if (!file.endsWith(".json") || file.endsWith(".tmp")) continue;
		try {
			const content = await fsp.readFile(path.join(dir, file), "utf8");
			const state = JSON.parse(content) as LoopState;
			results.push({
				id: state.id,
				status: state.status,
				objective: state.blueprint.objective,
				attempts: state.history.length,
				updatedAt: state.updatedAt,
			});
		} catch {
			// skip corrupt files
		}
	}
	return results.sort((a, b) => b.updatedAt - a.updatedAt);
}

export function newLoopId(): string {
	const ts = Date.now().toString(36);
	const rand = Math.random().toString(36).slice(2, 8);
	return `loop-${ts}-${rand}`;
}

export function newLoopState(blueprint: LoopBlueprint): LoopState {
	const now = Date.now();
	return {
		id: newLoopId(),
		blueprint,
		status: "running",
		startedAt: now,
		updatedAt: now,
		history: [],
		noProgressCount: 0,
	};
}
