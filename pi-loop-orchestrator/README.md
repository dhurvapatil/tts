# pi-loop-orchestrator

A [pi](https://pi.dev) extension that wraps **loop-engineering semantics** — objective, state, iteration prompt, verification, stop rules — around parallel multi-agent fan-out for build/test/fix cycles.

> Forked from [`pi-subagents`](https://github.com/mariozechner/pi-mono)'s spawn pattern. Adds the missing piece: a real verification gate (exit-code-based) wired into a state machine with stop rules and no-progress detection.

## Why

Existing multi-agent pi extensions spawn sub-agents in parallel but don't:

1. **Persist iteration state** between turns (so a long loop survives restarts)
2. **Define a done-state** as a verification command with a real exit code (not "the LLM thinks it's done")
3. **Enforce stop rules** (max iterations, max duration, no-progress detection)
4. **Parameterize iteration prompts** from prior iteration's reports + verification result

`pi-loop-orchestrator` adds all four on top of the spawn layer.

## Architecture

```
                ┌─────────────────────┐
                │  Orchestrator LLM   │  (the user-facing pi session)
                │  (root agent)       │
                └──────────┬──────────┘
                           │  calls tools
                           ▼
        ┌──────────────────────────────────────┐
        │ loop_design / loop_start /           │
        │ loop_run_iteration / loop_get_state  │
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │      Controller (this package)       │
        │                                      │
        │  state machine + stop rules +        │
        │  no-progress detection + reflection  │
        └──┬───────────────┬──────────────────┘
           │               │
   ┌───────▼──────┐   ┌────▼──────────────┐
   │ Fan-out layer│   │ Verification gate │
   │ (spawn.ts)   │   │ (runner.ts)       │
   └───────┬──────┘   └────┬──────────────┘
           │               │
   ┌───────▼───────────────▼──────────────┐
   │  N parallel `pi` subprocesses       │
   │  (implementer / tester / reviewer)  │
   │  + 1 verification command           │
   └─────────────────────────────────────┘
```

Each `loop_run_iteration` call is one iteration:
1. Load state from `~/.pi/agent/loops/<id>.json`
2. Apply stop rules (max iterations, max duration)
3. Build iteration prompts from blueprint templates, parameterized by last iteration's reports + verification
4. Run the fan-out (sequential steps, parallel variants per step)
5. Run the verification command (real shell, exit code is the ground truth)
6. Compute a recommendation (continue / stop / fail)
7. Persist the iteration, update state
8. Return `{passed, verification, recommendation, attempt, status}`

The orchestrator LLM calls `loop_run_iteration` repeatedly until `recommendation.action === "stop"`.

## Installation

### From a local path (development)

```bash
pi install /path/to/pi-loop-orchestrator
# or
pi install ./pi-loop-orchestrator
```

### From a git repo

```bash
pi install git:github.com/you/pi-loop-orchestrator
```

## Usage

### Quick start (auto-detected verify command)

Tell your pi agent what you want. It will design the loop and run it:

> "Get the test suite in this repo to pass."

The agent will:
1. Call `loop_design` with the objective
2. Call `loop_start` with the resulting blueprint
3. Call `loop_run_iteration` in a loop until the verification command passes

### Use a bundled example blueprint

```
Use the build-fix loop from loops/build-fix.yaml
```

The agent calls `loop_start(blueprintPath: "loops/build-fix.yaml")`.

### Parallel implementers

```
Try fixing this with the parallel-implementers loop
```

Spawns 3 implementer variants in parallel per iteration.

### Inspect or stop a loop

```
loop_get_state(loopId)    # show history
loop_stop(loopId, reason) # terminate
```

Or as a command:
```
/loop-roles   # list bundled agent roles
```

## Tools exposed to the LLM

| Tool | Purpose |
|------|---------|
| `loop_design` | Design a blueprint from an objective (auto-detects verify command from `package.json` / `Cargo.toml` / `pyproject.toml` / `go.mod` / etc.) |
| `loop_start` | Initialize a loop. Accepts a blueprint object, inline YAML, or a path to a `.yaml` file |
| `loop_run_iteration` | Run ONE iteration (fan-out + verify + recommend) |
| `loop_get_state` | Inspect a loop's history. Omit id to list all loops |
| `loop_stop` | Manually terminate a loop |

## Loop blueprint format

```yaml
# loops/build-fix.yaml
name: build-fix
objective: |
  The test suite exits 0 with no failures and the reviewer approves.

maxIterations: 8
maxDurationMinutes: 30
noProgressLimit: 3
verifyCommand: npm test
verifyTimeoutMs: 180000

fanOut:
  - role: implementer
    count: 1
    task: |
      You are the implementer in iteration {{attempt}}.
      Objective: {{objective}}
      {{#if focus}}Focus: {{focus}}{{/if}}
      ...

  - role: tester
    count: 1
    task: |
      You are the tester. Run `{{verifyCommand}}` and report.

  - role: reviewer
    count: 1
    task: |
      You are the reviewer. Run `git diff` and produce a verdict.
```

### Template variables

In any `task` field, you can use:

| Variable | Value |
|----------|-------|
| `{{attempt}}` | 1-based iteration number |
| `{{objective}}` | The loop's objective |
| `{{focus}}` | The current focus hint (user-supplied or auto-derived) |
| `{{lastReports}}` | Concatenated agent reports from the previous iteration |
| `{{lastVerification}}` | Previous verification's exit code, stderr tail, stdout tail |
| `{{verifyCommand}}` | The verification command (so agents can run it) |
| `{{variant}}` | For `count > 1`, the variant index (0-based) |
| `{{count}}` | For `count > 1`, the total number of variants |

Conditional blocks: `{{#if key}}...{{/if}}` (omitted if key is empty/null).

### Fan-out semantics

- Each entry in `fanOut[]` is a **step**. Steps run sequentially.
- Within a step, `count: N` spawns N parallel variants of the same role.
- Default `count: 1`.
- Example: `[implementer(3), tester, reviewer]` = 3 implementers in parallel, then 1 tester, then 1 reviewer.

## Bundled agent roles

| Role | Default model | Tools | Purpose |
|------|---------------|-------|---------|
| `implementer` | sonnet | read, edit, write, bash, grep, find, ls | Make code changes |
| `tester` | haiku | bash, read, grep, find | Run tests, report failures |
| `reviewer` | sonnet | read, grep, find, ls, bash | Review changes, judge quality |

Each role is a `.md` file in `agents/` with YAML frontmatter (`name`, `description`, `tools`, `model`) and a body that is the system prompt. Override any role by editing the `.md` file (e.g., change the model or add/remove tools).

## Stop rules

The controller enforces all of these:

1. **`maxIterations`** — stop after this many iterations
2. **`maxDurationMs`** — stop after this much wall time has elapsed
3. **`noProgressLimit`** — stop if the same failure pattern (exit code + top failing tests) repeats N times in a row
4. **Verification passed** — stop with `status: completed` if the verification command exits 0
5. **All agents produced no usable output** — stop with `status: failed` (suggests misconfiguration)

## State persistence

Each loop's state is stored at `~/.pi/agent/loops/<id>.json`. The state survives pi restarts. To resume after a restart, the orchestrator LLM calls `loop_get_state(loopId)` to see the history, then `loop_run_iteration(loopId, focus)` to continue.

## Layout

```
pi-loop-orchestrator/
├── package.json
├── index.ts                 # entry point, registers tools
├── README.md
├── agents/                  # bundled agent .md files
│   ├── implementer.md
│   ├── tester.md
│   └── reviewer.md
├── loops/                   # example loop blueprints
│   ├── build-fix.yaml
│   ├── parallel-implementers.yaml
│   └── review-only.yaml
└── src/
    ├── loop/
    │   ├── state.ts         # types + persistence
    │   ├── yaml.ts          # minimal YAML parser
    │   ├── designer.ts      # objective → blueprint
    │   └── controller.ts    # run_iteration, stop rules, reflection
    ├── agents/
    │   ├── roles.ts         # role registry, file resolution
    │   └── spawn.ts         # spawn pi subprocesses, run fan-out
    └── verify/
        └── runner.ts        # shell out to verify command, parse output
```

## How it's different from `pi-subagents`

| Capability | `pi-subagents` | `pi-loop-orchestrator` |
|------------|----------------|------------------------|
| Spawn child agents | yes | yes (same pattern) |
| Parallel fan-out | yes (one tool call) | yes (count=N per step, in blueprint) |
| State persistence | no | yes (`~/.pi/agent/loops/<id>.json`) |
| Stop rules | no | yes (max iter, max duration, no-progress) |
| Verification gate | no | yes (real shell command, exit-code-based) |
| Reflection / focus hints | no | yes (auto-derived from failure) |
| YAML blueprints | no | yes (bundled examples) |

## Limitations

- **Output truncated** to 30KB per agent report and 30KB per verify stream. Use the full `details` payload for the rest.
- **Same-cwd parallelism**: when `count > 1`, all variants run in the same working directory, so later variants see earlier variants' edits. For true isolation, you'd need git worktrees (not yet implemented — see `loops/parallel-implementers.yaml` for the discussion).
- **LLM judge still part of stop rule**: the reviewer agent's verdict is reported but the *hard* stop is the verification command's exit code. If you want a soft stop on reviewer blocks too, extend the `computeRecommendation` in `controller.ts`.
- **No editor integration**: this is a pure tool/command extension. There's no /tree or custom UI for loops. (Easy to add via `setWidget` if you want a live progress display.)

## Development

```bash
# Run from source
pi -e /path/to/pi-loop-orchestrator

# Then in pi, prompt it to run a loop
> Run the build-fix loop on this project
```

To extend, the most likely edit points are:
- `agents/*.md` — change a role's system prompt, model, or tools
- `loops/*.yaml` — change a blueprint's objective, fan-out, or stop rules
- `src/loop/controller.ts` — change the recommendation logic or no-progress heuristic
- `src/loop/designer.ts` — change the auto-detected verify command or fan-out defaults
