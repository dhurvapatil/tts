/**
 * Loop blueprint designer.
 *
 * Takes an objective and produces a LoopBlueprint. The designer is intentionally
 * heuristic-only — it doesn't call an LLM. The point of the blueprint is to be
 * a *starting point* that the orchestrator LLM can review and adjust via
 * loop_start's overrides before committing.
 *
 * Detection sources for verifyCommand:
 *   - package.json scripts.test   -> "npm test"
 *   - pyproject.toml              -> "pytest" or "uv run pytest"
 *   - Cargo.toml                  -> "cargo test"
 *   - go.mod                      -> "go test ./..."
 *   - Makefile (test target)      -> "make test"
 *   - deno.json                   -> "deno test"
 *   - bun.lockb or bunfig.toml    -> "bun test"
 *   - vitest.config.*             -> "npx vitest run"
 *   - jest.config.*               -> "npx jest"
 *
 * Fan-out shape:
 *   - "default"        implementer -> tester -> reviewer
 *   - "parallel-impl"  N implementers (count=3) -> tester -> reviewer
 *   - "minimal"        implementer -> tester
 *   - "review-only"    reviewer
 */

import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { LoopBlueprint, FanOutStep } from "./state.ts";

export interface DesignHints {
	verifyCommand?: string;
	maxIterations?: number;
	maxDurationMinutes?: number;
	noProgressLimit?: number;
	fanOut?: "default" | "parallel-impl" | "minimal" | "review-only" | string;
	projectType?: string; // override auto-detection
}

export async function designBlueprint(
	objective: string,
	hints: DesignHints | undefined,
	cwd: string,
): Promise<LoopBlueprint> {
	const projectType = hints?.projectType ?? (await detectProjectType(cwd));
	const verifyCommand = hints?.verifyCommand ?? (await detectVerifyCommand(cwd, projectType));

	const fanOut = designFanOut(objective, hints?.fanOut ?? "default");

	return {
		name: `loop-${Date.now().toString(36)}`,
		description: `Auto-designed loop for: ${truncate(objective, 80)}`,
		objective: objective.trim(),
		fanOut,
		stopRules: {
			maxIterations: hints?.maxIterations ?? 8,
			maxDurationMs: (hints?.maxDurationMinutes ?? 30) * 60 * 1000,
			noProgressLimit: hints?.noProgressLimit ?? 3,
			verifyCommand,
			verifyTimeoutMs: 180_000,
			verifyCwd: cwd,
		},
	};
}

async function detectProjectType(cwd: string): Promise<string> {
	const checks: Array<[string, string]> = [
		["package.json", "node"],
		["pyproject.toml", "python"],
		["Cargo.toml", "rust"],
		["go.mod", "go"],
		["deno.json", "deno"],
		["bunfig.toml", "bun"],
		["Gemfile", "ruby"],
		["build.gradle", "gradle"],
		["pom.xml", "maven"],
		["mix.exs", "elixir"],
	];
	for (const [file, type] of checks) {
		try {
			await fs.access(path.join(cwd, file));
			return type;
		} catch {
			// not this type
		}
	}
	return "unknown";
}

async function detectVerifyCommand(cwd: string, projectType: string): Promise<string> {
	switch (projectType) {
		case "node": {
			try {
				const pkg = JSON.parse(await fs.readFile(path.join(cwd, "package.json"), "utf8"));
				const scripts = pkg.scripts ?? {};
				if (scripts.test) return "npm test";
			} catch {
				// ignore
			}
			return "npm test";
		}
		case "python": {
			try {
				const content = await fs.readFile(path.join(cwd, "pyproject.toml"), "utf8");
				if (/\bpytest\b/.test(content)) return "pytest";
				if (/\bunittest\b/.test(content)) return "python -m unittest";
			} catch {
				// ignore
			}
			return "pytest";
		}
		case "rust":
			return "cargo test";
		case "go":
			return "go test ./...";
		case "deno":
			return "deno test";
		case "bun":
			return "bun test";
		case "ruby":
			return "bundle exec rspec || bundle exec rake test";
		case "gradle":
			return "./gradlew test";
		case "maven":
			return "mvn test";
		case "elixir":
			return "mix test";
		default:
			return "echo 'no verify command detected; pass verifyCommand in hints'";
	}
}

function designFanOut(objective: string, mode: string): FanOutStep[] {
	const lower = objective.toLowerCase();
	const isReviewOnly = mode === "review-only" || /review|audit|inspect/.test(lower) && !/fix|build|implement|refactor/.test(lower);
	if (isReviewOnly) {
		return [
			{
				role: "reviewer",
				count: 1,
				task: defaultReviewerTask(),
			},
		];
	}

	switch (mode) {
		case "parallel-impl":
			return [
				{ role: "implementer", count: 3, task: defaultImplementerTask() },
				{ role: "tester", count: 1, task: defaultTesterTask() },
				{ role: "reviewer", count: 1, task: defaultReviewerTask() },
			];
		case "minimal":
			return [
				{ role: "implementer", count: 1, task: defaultImplementerTask() },
				{ role: "tester", count: 1, task: defaultTesterTask() },
			];
		case "default":
		default:
			return [
				{ role: "implementer", count: 1, task: defaultImplementerTask() },
				{ role: "tester", count: 1, task: defaultTesterTask() },
				{ role: "reviewer", count: 1, task: defaultReviewerTask() },
			];
	}
}

function defaultImplementerTask(): string {
	return [
		"You are the **implementer** in iteration {{attempt}} of a build/test/fix loop.",
		"{{#if variant}}You are variant {{variant}} — try a meaningfully different approach than other variants.{{/if}}",
		"{{#if focus}}**This iteration's focus:** {{focus}}{{/if}}",
		"",
		"**Objective (done-state):** {{objective}}",
		"",
		"{{#if lastReports}}**Previous iteration's agent reports:**",
		"{{lastReports}}{{/if}}",
		"",
		"{{#if lastVerification}}**Previous verification result:**",
		"{{lastVerification}}{{/if}}",
		"",
		"Your job:",
		"1. Read the relevant files. Start from the failing test output or focus hint.",
		"2. Form a hypothesis about the smallest change that moves toward the objective.",
		"3. Make targeted edits with the edit/write tools. Do not refactor unrelated code.",
		"4. If a previous approach failed, try a different angle — do not repeat the same edit.",
		"",
		"**End your response with this exact block:**",
		"```",
		"CHANGED_FILES: <comma-separated paths>",
		"SUMMARY: <one-sentence description of the change>",
		"ROOT_CAUSE: <one-sentence root cause or 'unknown'>",
		"CONFIDENCE: low | medium | high",
		"```",
	].join("\n");
}

function defaultTesterTask(): string {
	return [
		"You are the **tester** in iteration {{attempt}} of a build/test/fix loop.",
		"",
		"**Objective (done-state):** {{objective}}",
		"",
		"Run the verification command and report structured findings. Do not modify code.",
		"",
		"Use the bash tool to run: `{{verifyCommand}}`",
		"(If the project has a longer-running test script, you may increase the timeout.)",
		"",
		"**Report format (use this exact template at the end):**",
		"```",
		"STATUS: PASS | FAIL",
		"DURATION: <seconds>s",
		"EXIT_CODE: <number>",
		"FAILING_TESTS:",
		"  - <name> in <path>",
		"    Error: <one-line error>",
		"TOTAL_FAILURES: <count>",
		"```",
	].join("\n");
}

function defaultReviewerTask(): string {
	return [
		"You are the **reviewer** in iteration {{attempt}} of a build/test/fix loop.",
		"",
		"**Objective (done-state):** {{objective}}",
		"",
		"{{#if lastReports}}**Implementer's report from this iteration:**",
		"{{lastReports}}{{/if}}",
		"",
		"Review the changes made this iteration:",
		"1. Run `git diff` (or `git diff --stat` for a summary) to see what changed.",
		"2. For each change, check: does it actually move toward the objective?",
		"   Are there obvious bugs, race conditions, security issues, or scope creep?",
		"3. Form a verdict. Only block if there is a real, demonstrable problem.",
		"",
		"**Report format (use this exact template at the end):**",
		"```",
		"VERDICT: APPROVE | BLOCK",
		"BLOCKERS:",
		"  - <one-line description of each blocking issue>",
		"NOTES:",
		"  - <one-line non-blocking concern>",
		"```",
	].join("\n");
}

function truncate(s: string, n: number): string {
	return s.length <= n ? s : s.slice(0, n - 3) + "...";
}
