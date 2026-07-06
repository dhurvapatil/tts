/**
 * Smoke test: parse the bundled loop YAML files and verify the shape.
 * Exercises the designer, the YAML parser, and the role registry.
 *
 * Uses jiti (bundled with pi-coding-agent) to load .ts sources directly.
 *
 * Run:  node scripts/smoke-test.mjs
 */

import { readFile, mkdtemp, writeFile } from "node:fs/promises";
import * as path from "node:path";
import * as os from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";

// Use jiti to load the project's TypeScript sources.
const jitiUrl = pathToFileURL(path.resolve("node_modules/jiti/lib/jiti.mjs")).href;
const { createJiti } = await import(jitiUrl);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PACKAGE_ROOT = path.resolve(__dirname, "..");

const jiti = createJiti(import.meta.url, {
	esmResolve: true,
	interopDefault: true,
});

const yamlMod = await jiti.import(path.join(PACKAGE_ROOT, "src/loop/yaml.ts"));
const designerMod = await jiti.import(path.join(PACKAGE_ROOT, "src/loop/designer.ts"));
const rolesMod = await jiti.import(path.join(PACKAGE_ROOT, "src/agents/roles.ts"));

const { parseBlueprintYaml, normalizeBlueprint } = yamlMod;
const { designBlueprint } = designerMod;
const { listAvailableRoles, loadAgentDef } = rolesMod;

const tests = [];
let passed = 0;
let failed = 0;

function test(name, fn) {
	tests.push({ name, fn });
}

async function runTests() {
	for (const t of tests) {
		try {
			await t.fn();
			console.log(`  \u2713 ${t.name}`);
			passed++;
		} catch (err) {
			console.log(`  \u2717 ${t.name}`);
			console.log(`      ${err.message}`);
			if (err.stack) console.log(`      ${err.stack.split("\n").slice(1, 4).join("\n      ")}`);
			failed++;
		}
	}
	console.log(`\n${passed} passed, ${failed} failed`);
	process.exit(failed > 0 ? 1 : 0);
}

test("parse build-fix.yaml", async () => {
	const content = await readFile(path.join(PACKAGE_ROOT, "loops/build-fix.yaml"), "utf8");
	const parsed = parseBlueprintYaml(content);
	const bp = normalizeBlueprint(parsed);

	assertEq(bp.objective.length > 0, true, "objective should be non-empty");
	assertEq(Array.isArray(bp.fanOut), true, "fanOut should be array");
	assertEq(bp.fanOut.length, 3, "expected 3 fanOut steps");
	assertEq(bp.fanOut[0].role, "implementer", "first step should be implementer");
	assertEq(bp.fanOut[1].role, "tester", "second step should be tester");
	assertEq(bp.fanOut[2].role, "reviewer", "third step should be reviewer");
	assertEq(bp.stopRules.maxIterations, 8, "maxIterations");
	assertEq(bp.stopRules.verifyCommand, "npm test", "verifyCommand");
	assertEq(typeof bp.fanOut[0].task, "string", "implementer task type");
	assert(bp.fanOut[0].task.includes("{{attempt}}"), "implementer task should reference {{attempt}}");
});

test("parse parallel-implementers.yaml", async () => {
	const content = await readFile(path.join(PACKAGE_ROOT, "loops/parallel-implementers.yaml"), "utf8");
	const bp = normalizeBlueprint(parseBlueprintYaml(content));
	assertEq(bp.fanOut.length, 3, "3 steps");
	assertEq(bp.fanOut[0].count, 3, "first step count 3");
});

test("parse review-only.yaml", async () => {
	const content = await readFile(path.join(PACKAGE_ROOT, "loops/review-only.yaml"), "utf8");
	const bp = normalizeBlueprint(parseBlueprintYaml(content));
	assertEq(bp.fanOut.length, 1, "1 step");
	assertEq(bp.fanOut[0].role, "reviewer", "reviewer");
	assertEq(bp.stopRules.maxIterations, 1, "single-iteration");
});

test("list bundled agent roles", async () => {
	const roles = await listAvailableRoles();
	assert(roles.includes("implementer"), `roles missing implementer: ${roles.join(", ")}`);
	assert(roles.includes("tester"), `roles missing tester: ${roles.join(", ")}`);
	assert(roles.includes("reviewer"), `roles missing reviewer: ${roles.join(", ")}`);
});

test("load each agent .md and parse frontmatter", async () => {
	for (const name of ["implementer", "tester", "reviewer"]) {
		const def = await loadAgentDef(name);
		assert(def !== null, `${name} should load`);
		assertEq(def.name, name, `${name} name`);
		assert(def.systemPrompt.length > 100, `${name} should have non-trivial prompt`);
	}
	const tester = await loadAgentDef("tester");
	assertEq(tester.model, "haiku", "tester should default to haiku");
	const implementer = await loadAgentDef("implementer");
	assertEq(implementer.model, "sonnet", "implementer should default to sonnet");
});

test("design blueprint with auto-detected verify command", async () => {
	const tmpDir = await mkdtemp(path.join(os.tmpdir(), "pi-loop-smoke-"));
	await writeFile(path.join(tmpDir, "package.json"), JSON.stringify({
		name: "fake",
		scripts: { test: "echo tested" },
	}));
	const bp = await designBlueprint("make tests pass", undefined, tmpDir);
	assertEq(bp.objective, "make tests pass", "objective");
	assertEq(bp.stopRules.verifyCommand, "npm test", "auto-detect npm test");
	assert(bp.fanOut.length >= 2, "should have implementer+tester minimum");
});

test("design blueprint with explicit verify command override", async () => {
	const tmpDir = await mkdtemp(path.join(os.tmpdir(), "pi-loop-smoke-"));
	const bp = await designBlueprint("run pytest", { verifyCommand: "pytest -x" }, tmpDir);
	assertEq(bp.stopRules.verifyCommand, "pytest -x", "explicit override");
});

test("design blueprint with parallel-impl fan-out", async () => {
	const tmpDir = await mkdtemp(path.join(os.tmpdir(), "pi-loop-smoke-"));
	const bp = await designBlueprint("try several approaches", { fanOut: "parallel-impl" }, tmpDir);
	assertEq(bp.fanOut[0].role, "implementer", "first step is implementer");
	assertEq(bp.fanOut[0].count, 3, "parallel-impl should have count=3");
});

test("YAML parser: block scalars (|)", () => {
	const yaml = "key: |\n  line one\n  line two\n    indented\nnext: done\n";
	const parsed = parseBlueprintYaml(yaml);
	assertEq(parsed.key, "line one\nline two\n  indented", `got: ${JSON.stringify(parsed.key)}`);
	assertEq(parsed.next, "done", "next should follow");
});

test("YAML parser: folded scalars (>)", () => {
	const yaml = "key: >\n  line one\n  line two\nnext: done\n";
	const parsed = parseBlueprintYaml(yaml);
	assertEq(parsed.key, "line one line two", `got: ${JSON.stringify(parsed.key)}`);
});

test("YAML parser: scalars (number/bool/null)", () => {
	const parsed = parseBlueprintYaml(`
int: 42
float: 3.14
yes: yes
no: no
nothing: null
quoted: "with spaces"
`);
	assertEq(parsed.int, 42, "int");
	assertEq(parsed.float, 3.14, "float");
	assertEq(parsed.yes, true, "yes");
	assertEq(parsed.no, false, "no");
	assertEq(parsed.nothing, null, "null");
	assertEq(parsed.quoted, "with spaces", "quoted");
});

test("YAML parser: comments", () => {
	const parsed = parseBlueprintYaml(`
key: value  # trailing comment
# full-line comment
other: 42  # also commented
`);
	assertEq(parsed.key, "value", "value");
	assertEq(parsed.other, 42, "other");
});

test("YAML parser: nested sequences of mappings", () => {
	const parsed = parseBlueprintYaml(`
items:
  - name: a
    value: 1
  - name: b
    value: 2
`);
	assertEq(parsed.items.length, 2, "items length");
	assertEq(parsed.items[0].name, "a", "first name");
	assertEq(parsed.items[1].value, 2, "second value");
});

test("YAML parser: deep nesting", () => {
	const parsed = parseBlueprintYaml(`
outer:
  inner:
    deeper:
      key: value
`);
	assertEq(parsed.outer.inner.deeper.key, "value", "deep value");
});

function assert(cond, msg) {
	if (!cond) throw new Error(msg);
}

function assertEq(actual, expected, msg) {
	if (actual !== expected) {
		throw new Error(`${msg}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
	}
}

runTests();
