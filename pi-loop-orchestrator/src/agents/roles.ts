/**
 * Role registry. Maps role names to bundled agent definition files.
 *
 * The agents/ directory at the package root holds .md files with YAML
 * frontmatter (name, description, tools, model) followed by the
 * system prompt body. This mirrors the subagent extension's format
 * so users can add custom roles by dropping new .md files in the dir.
 */

import * as fs from "node:fs/promises";
import * as fsSync from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export interface AgentDef {
	name: string;
	description: string;
	tools?: string[];
	model?: string;
	systemPrompt: string;
	filePath: string;
}

/**
 * Walk up from the current file to find the package root (the directory
 * containing our package.json). Works whether the extension is loaded
 * from source or installed via pi install.
 */
export function findPackageRoot(): string {
	let current = __dirname;
	for (let i = 0; i < 8; i++) {
		const pkgPath = path.join(current, "package.json");
		try {
			const pkg = JSON.parse(fsSync.readFileSync(pkgPath, "utf8"));
			if (pkg.name === "pi-loop-orchestrator") return current;
		} catch {
			// not us, keep walking
		}
		const parent = path.dirname(current);
		if (parent === current) break;
		current = parent;
	}
	// Fallback: walk up from src/agents/ to package root
	return path.resolve(__dirname, "..", "..");
}

export function getAgentsDir(): string {
	return path.join(findPackageRoot(), "agents");
}

export async function loadAgentDef(roleName: string): Promise<AgentDef | null> {
	const dir = getAgentsDir();

	// 1. Exact match: <role>.md
	const exact = path.join(dir, `${roleName}.md`);
	if (await fileExists(exact)) {
		const content = await fs.readFile(exact, "utf8");
		return parseAgentDef(content, exact);
	}

	// 2. Try without trailing "er" normalization: implementer -> implement
	const trimmed = roleName.replace(/(er|or)$/, "");
	if (trimmed !== roleName) {
		const alt = path.join(dir, `${trimmed}.md`);
		if (await fileExists(alt)) {
			const content = await fs.readFile(alt, "utf8");
			return parseAgentDef(content, alt);
		}
	}

	// 3. Scan dir for a file whose frontmatter name matches
	try {
		const files = await fs.readdir(dir);
		for (const file of files) {
			if (!file.endsWith(".md")) continue;
			const filePath = path.join(dir, file);
			const content = await fs.readFile(filePath, "utf8");
			const def = parseAgentDef(content, filePath);
			if (def.name === roleName) return def;
		}
	} catch {
		// dir not readable
	}

	return null;
}

export async function listAvailableRoles(): Promise<string[]> {
	const dir = getAgentsDir();
	try {
		const files = await fs.readdir(dir);
		const roles: string[] = [];
		for (const file of files) {
			if (!file.endsWith(".md")) continue;
			const filePath = path.join(dir, file);
			try {
				const content = await fs.readFile(filePath, "utf8");
				const def = parseAgentDef(content, filePath);
				roles.push(def.name);
			} catch {
				// skip
			}
		}
		return roles;
	} catch {
		return [];
	}
}

function parseAgentDef(content: string, filePath: string): AgentDef {
	const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
	if (!match) {
		return {
			name: path.basename(filePath, ".md"),
			description: "",
			systemPrompt: content,
			filePath,
		};
	}
	const frontmatter = match[1];
	const body = match[2];
	const meta: Record<string, string> = {};
	for (const line of frontmatter.split(/\r?\n/)) {
		const m = line.match(/^([\w-]+):\s*(.*)$/);
		if (m) {
			let val = m[2].trim();
			// Strip surrounding quotes
			if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
				val = val.slice(1, -1);
			}
			meta[m[1]] = val;
		}
	}
	return {
		name: meta.name ?? path.basename(filePath, ".md"),
		description: meta.description ?? "",
		tools: meta.tools
			?.split(",")
			.map((t) => t.trim())
			.filter(Boolean),
		model: meta.model,
		systemPrompt: body.trim(),
		filePath,
	};
}

async function fileExists(p: string): Promise<boolean> {
	try {
		await fs.access(p);
		return true;
	} catch {
		return false;
	}
}
