/**
 * pi-loop-orchestrator
 *
 * Wraps loop-engineering semantics (objective, state, iteration prompt,
 * verification, stop rules) around parallel multi-agent fan-out for
 * build/test/fix cycles. See README.md for the full architecture.
 *
 * Public tools exposed to the LLM:
 *   - loop_design         produce a blueprint from an objective
 *   - loop_start          initialize a loop (accepts object, YAML, or path)
 *   - loop_run_iteration  run one iteration (fan-out + verify + recommend)
 *   - loop_get_state      inspect a loop's history
 *   - loop_stop           manually terminate a loop
 */

import * as fsp from "node:fs/promises";
import * as path from "node:path";
import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { designBlueprint, type DesignHints } from "./src/loop/designer.ts";
import { normalizeBlueprint, parseBlueprintYaml } from "./src/loop/yaml.ts";
import { startLoop, runIteration, getLoopState, listLoops, stopLoop, type IterationResult } from "./src/loop/controller.ts";
import { listAvailableRoles } from "./src/agents/roles.ts";

const LOOP_GUIDANCE = `## Loop Orchestrator

You can drive iterative build/test/fix cycles by calling these tools in order:

1. **loop_design(objective, hints?)** — Produce a loop blueprint. Returns the proposed verification command, fan-out shape, and stop rules. Review before starting.
2. **loop_start(blueprint | blueprintYaml | blueprintPath?)** — Initialize loop state. Accepts either a designed blueprint object, an inline YAML string, or a path to a \`.yaml\` file (e.g. \`loops/build-fix.yaml\`). Returns a \`loopId\`.
3. **loop_run_iteration(loopId, focus?)** — Run ONE iteration:
   - Reads state, builds iteration prompts from blueprint templates, parameterized by current state.
   - Spawns the fan-out wave (implementer/tester/reviewer, with parallel variants per step).
   - Runs the verification command — exit code is the **only** objective ground truth.
   - Persists the iteration, returns \`{ passed, verification, recommendation, attempt, status }\`.
4. **loop_get_state(loopId?)** — Inspect history. Omit id to list all loops.
5. **loop_stop(loopId, reason)** — Manually terminate a loop.

Iterate \`loop_run_iteration\` until \`recommendation.action === "stop"\`. Each call is one iteration. Don't try to batch — call once, read the result, decide, call again.

The verification gate matters: tests must pass. If a previous iteration's approach is failing repeatedly (same exit code + same failing tests), the loop will auto-stop after \`noProgressLimit\` iterations.`;

export default function (pi: ExtensionAPI) {
	// Inject loop orchestrator guidance into the system prompt once.
	pi.on("before_agent_start", async (event) => {
		if (event.systemPrompt.includes("## Loop Orchestrator")) return;
		return {
			systemPrompt: event.systemPrompt + "\n\n" + LOOP_GUIDANCE,
		};
	});

	pi.registerTool({
		name: "loop_design",
		label: "Design Loop",
		description:
			"Design a loop blueprint for a build/test/fix objective. Returns a LoopBlueprint with the proposed verification command, fan-out shape, and stop rules. Review the result before calling loop_start.",
		promptSnippet: "Design a loop blueprint from an objective (returns verification command, fan-out, stop rules).",
		parameters: Type.Object({
			objective: Type.String({
				description: "The done-state the loop is trying to achieve. Frame as a state, not an activity (e.g., 'npm test exits 0 with no failures' rather than 'make the tests pass').",
			}),
			hints: Type.Optional(
				Type.Object({
					verifyCommand: Type.Optional(
						Type.String({ description: "Override the auto-detected verification command (e.g., 'npm test', 'pytest', 'cargo test')." }),
					),
					maxIterations: Type.Optional(Type.Number({ description: "Override default max iterations (8)." })),
					maxDurationMinutes: Type.Optional(Type.Number({ description: "Override default max duration in minutes (30)." })),
					noProgressLimit: Type.Optional(Type.Number({ description: "Stop if the same failure pattern repeats this many times in a row (3)." })),
					fanOut: Type.Optional(
						Type.String({
							description: "Fan-out shape: 'default' (implementer->tester->reviewer), 'parallel-impl' (3 implementers in parallel), 'minimal' (implementer->tester), 'review-only'.",
						}),
					),
				}),
			),
		}),
		async execute(_id, params, _signal, _onUpdate, ctx) {
			const blueprint = await designBlueprint(params.objective, params.hints as DesignHints | undefined, ctx.cwd);
			return {
				content: [{ type: "text", text: JSON.stringify(blueprint, null, 2) }],
				details: { blueprint },
			};
		},
	});

	pi.registerTool({
		name: "loop_start",
		label: "Start Loop",
		description:
			"Start a loop with the given blueprint. Accepts one of: a blueprint object (from loop_design), an inline YAML string, or a path to a .yaml file. Returns a loopId for subsequent calls.",
		promptSnippet: "Initialize a loop with a blueprint; returns loopId for iteration calls.",
		parameters: Type.Object({
			blueprint: Type.Optional(Type.Any({ description: "Blueprint object (e.g., from loop_design output)." })),
			blueprintYaml: Type.Optional(Type.String({ description: "Inline YAML blueprint." })),
			blueprintPath: Type.Optional(Type.String({ description: "Path to a .yaml blueprint file, relative to cwd or absolute." })),
		}),
		async execute(_id, params, _signal, _onUpdate, ctx) {
			let blueprint: any;
			if (params.blueprint) {
				blueprint = params.blueprint;
			} else if (params.blueprintYaml) {
				blueprint = parseBlueprintYaml(params.blueprintYaml);
			} else if (params.blueprintPath) {
				const resolved = path.isAbsolute(params.blueprintPath)
					? params.blueprintPath
					: path.resolve(ctx.cwd, params.blueprintPath);
				const content = await fsp.readFile(resolved, "utf8");
				blueprint = parseBlueprintYaml(content);
			} else {
				return {
					content: [{ type: "text", text: "Error: provide one of `blueprint`, `blueprintYaml`, or `blueprintPath`." }],
					details: {},
					isError: true,
				};
			}

			blueprint = normalizeBlueprint(blueprint);
			validateBlueprint(blueprint);

			const id = await startLoop(blueprint, ctx.cwd, pi);
			return {
				content: [
					{
						type: "text",
						text: `Loop started: ${id}\n\n${JSON.stringify(blueprint, null, 2)}`,
					},
				],
				details: { loopId: id, blueprint },
			};
		},
	});

	pi.registerTool({
		name: "loop_run_iteration",
		label: "Run Loop Iteration",
		description:
			"Run ONE iteration of the loop: build prompts from state, spawn the fan-out wave, run the verification command, update state, return pass/fail + recommendation. Call repeatedly until recommendation.action === 'stop'.",
		promptSnippet: "Run one iteration: fan-out agents, run verification, return pass/fail + next action.",
		parameters: Type.Object({
			loopId: Type.String({ description: "The loop id returned by loop_start." }),
			focus: Type.Optional(
				Type.String({
					description:
						"Optional focus hint for this iteration's prompts. Overrides the auto-derived focus from the previous failure. Example: 'try a different algorithm' or 'fix the off-by-one in parseInt'.",
				}),
			),
		}),
		async execute(_id, params, signal, onUpdate, ctx) {
			const result = await runIteration(params.loopId, params.focus, ctx.cwd, pi, signal);

			// Stream a status line while running (cosmetic, only if onUpdate supported)
			if (onUpdate) {
				onUpdate({
					content: [
						{
							type: "text",
							text: `Loop ${params.loopId} attempt ${result.attempt}: ${result.recommendation.action}${result.passed ? " (verification passed)" : ""}`,
						},
					],
					details: result as any,
				});
			}

			const summaryText = formatIterationSummary(result);
			return {
				content: [{ type: "text", text: summaryText }],
				details: result as any,
				isError: result.recommendation.action === "fail",
			};
		},
	});

	pi.registerTool({
		name: "loop_get_state",
		label: "Get Loop State",
		description: "Get the current state of a loop, including iteration history. Pass no loopId to list all loops.",
		promptSnippet: "Inspect a loop's state; omit id to list all loops.",
		parameters: Type.Object({
			loopId: Type.Optional(Type.String({ description: "Loop id to inspect. Omit to list all." })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			if (!params.loopId) {
				const loops = await listLoops();
				return {
					content: [{ type: "text", text: JSON.stringify(loops, null, 2) }],
					details: { loops },
				};
			}
			const state = await getLoopState(params.loopId);
			return {
				content: [{ type: "text", text: JSON.stringify(state, null, 2) }],
				details: state as any,
			};
		},
	});

	pi.registerTool({
		name: "loop_stop",
		label: "Stop Loop",
		description: "Manually stop a loop. Marks the loop state as 'stopped' and records the reason.",
		promptSnippet: "Manually stop a running loop.",
		parameters: Type.Object({
			loopId: Type.String({ description: "Loop id to stop." }),
			reason: Type.String({ description: "Why the loop is being stopped (free-form, recorded in state)." }),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const state = await stopLoop(params.loopId, params.reason);
			return {
				content: [{ type: "text", text: `Loop ${params.loopId} stopped: ${params.reason}\n\nFinal status: ${state.status}` }],
				details: state as any,
			};
		},
	});

	// Optional: command to show the available roles.
	pi.registerCommand("loop-roles", {
		description: "List the agent roles bundled with pi-loop-orchestrator",
		handler: async (_args, ctx) => {
			const roles = await listAvailableRoles();
			if (roles.length === 0) {
				ctx.ui.notify("No agent roles found in the bundled agents/ directory.", "warning");
				return;
			}
			ctx.ui.notify(`Available roles: ${roles.join(", ")}`, "info");
		},
	});
}

function validateBlueprint(bp: any): void {
	if (!bp || typeof bp !== "object") {
		throw new Error("blueprint must be an object");
	}
	if (typeof bp.objective !== "string" || bp.objective.trim() === "") {
		throw new Error("blueprint.objective must be a non-empty string (the done-state)");
	}
	if (!Array.isArray(bp.fanOut) || bp.fanOut.length === 0) {
		throw new Error("blueprint.fanOut must be a non-empty array of steps");
	}
	for (const step of bp.fanOut) {
		if (typeof step.role !== "string" || step.role === "") {
			throw new Error("each fanOut step must have a non-empty `role`");
		}
		if (typeof step.task !== "string" || step.task === "") {
			throw new Error(`fanOut step for role '${step.role}' must have a non-empty \`task\` template`);
		}
	}
	const sr = bp.stopRules ?? {};
	if (typeof sr.verifyCommand !== "string" || sr.verifyCommand === "") {
		throw new Error("blueprint.stopRules.verifyCommand is required");
	}
	if (typeof sr.maxIterations !== "number" || sr.maxIterations <= 0) {
		throw new Error("blueprint.stopRules.maxIterations must be a positive number");
	}
}

function formatIterationSummary(r: IterationResult): string {
	const lines: string[] = [];
	lines.push(`# Loop ${r.loopId} — attempt ${r.attempt}`);
	lines.push("");
	lines.push(`**Status:** ${r.status}`);
	lines.push(`**Verification passed:** ${r.passed ? "yes" : "no"}`);
	lines.push(`**Recommendation:** ${r.recommendation.action} — ${r.recommendation.reason}`);
	if (r.recommendation.suggestedFocus) {
		lines.push(`**Suggested focus:** ${r.recommendation.suggestedFocus}`);
	}
	lines.push(`**Duration:** ${r.durationMs}ms`);
	lines.push("");
	lines.push(`## Verification`);
	lines.push("```");
	lines.push(`command: ${r.verification.command}`);
	lines.push(`exit: ${r.verification.exitCode}`);
	lines.push(`timedOut: ${r.verification.timedOut}`);
	lines.push(`duration: ${r.verification.durationMs}ms`);
	lines.push("```");
	if (r.verification.stderr) {
		lines.push("");
		lines.push("**stderr (tail):**");
		lines.push("```");
		lines.push(tail(r.verification.stderr, 30));
		lines.push("```");
	}
	if (r.verification.stdout) {
		lines.push("");
		lines.push("**stdout (tail):**");
		lines.push("```");
		lines.push(tail(r.verification.stdout, 15));
		lines.push("```");
	}
	lines.push("");
	lines.push(`## Agent reports (${r.agentReports.length})`);
	for (const rep of r.agentReports) {
		const tag = rep.variant !== undefined && rep.variant > 0 ? ` (variant ${rep.variant})` : "";
		lines.push(`- **${rep.role}${tag}** — exit=${rep.exitCode}${rep.stopReason ? ` (${rep.stopReason})` : ""}`);
	}
	return lines.join("\n");
}

function tail(s: string, n: number): string {
	const lines = s.split("\n");
	return lines.slice(-n).join("\n");
}
