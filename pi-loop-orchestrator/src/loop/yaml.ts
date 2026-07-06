/**
 * Minimal YAML subset parser for loop blueprints.
 *
 * Supports: top-level mapping, nested mappings, sequences of mappings,
 * scalars (string/number/bool/null), block scalars (| literal, > folded),
 * # comments. Does NOT support: anchors/aliases, flow style ({}, []),
 * multi-doc (---), tags.
 *
 * This is enough for the loop blueprint format and avoids pulling in
 * a full YAML library. If you outgrow it, swap in js-yaml.
 */

export class YamlParseError extends Error {
	constructor(message: string, public line?: number) {
		super(line !== undefined ? `YAML parse error (line ${line}): ${message}` : `YAML parse error: ${message}`);
	}
}

interface Line {
	indent: number;
	content: string; // trimmed (used for key matching, list detection)
	raw: string; // original text after comment-strip (used for block scalars)
}

export function parseBlueprintYaml(input: string): any {
	return new YamlParser(input).parse();
}

class YamlParser {
	private lines: Line[] = [];
	private pos = 0;

	constructor(text: string) {
		this.lines = this.preprocess(text);
	}

	private preprocess(text: string): Line[] {
		const result: Line[] = [];
		const rawLines = text.split(/\r?\n/);
		for (const raw of rawLines) {
			const stripped = this.stripComment(raw);
			if (stripped.trim() === "") continue;
			const indent = stripped.length - stripped.trimStart().length;
			const content = stripped.trimStart();
			if (content === "") continue;
			result.push({ indent, content, raw: stripped });
		}
		return result;
	}

	private stripComment(line: string): string {
		let inString = false;
		let stringChar = "";
		for (let i = 0; i < line.length; i++) {
			const c = line[i];
			if (inString) {
				if (c === stringChar && line[i - 1] !== "\\") inString = false;
				continue;
			}
			if (c === '"' || c === "'") {
				inString = true;
				stringChar = c;
				continue;
			}
			if (c === "#") {
				return line.substring(0, i).replace(/\s+$/, "");
			}
		}
		return line.replace(/\s+$/, "");
	}

	parse(): any {
		if (this.lines.length === 0) return null;
		return this.parseNode(0);
	}

	private parseNode(minIndent: number): any {
		const first = this.lines[this.pos];
		if (!first || first.indent < minIndent) return undefined;
		if (first.content.startsWith("- ")) {
			return this.parseList(first.indent);
		}
		return this.parseMapping(first.indent);
	}

	private parseMapping(indent: number): Record<string, any> {
		const result: Record<string, any> = {};
		while (this.pos < this.lines.length) {
			const line = this.lines[this.pos];
			if (line.indent < indent) break;
			// A list item at the same indent ends the mapping.
			if (line.indent === indent && line.content.startsWith("- ")) break;

			// Deeper-indented lines are part of the previous key's value
			// (which would have been handled by recursion in the value
			// branch). If we somehow get here, treat them as a nested mapping.
			if (line.indent > indent) {
				const nested = this.parseMapping(line.indent);
				for (const k of Object.keys(nested)) {
					if (k in result) {
						throw new YamlParseError(
							`duplicate key '${k}' at line ${this.pos + 1}`,
							this.pos + 1,
						);
					}
					result[k] = nested[k];
				}
				continue;
			}

			const m = line.content.match(/^([^:]+):\s*(.*)$/);
			if (!m) {
				throw new YamlParseError(`expected 'key: value', got: ${line.content}`, this.pos + 1);
			}
			const key = m[1].trim();
			const valuePart = m[2];
			this.pos++;

			if (valuePart === "" || valuePart === undefined) {
				const next = this.lines[this.pos];
				if (!next || next.indent <= indent) {
					result[key] = null;
				} else if (next.content.startsWith("- ")) {
					result[key] = this.parseList(next.indent);
				} else {
					result[key] = this.parseMapping(next.indent);
				}
			} else if (valuePart === "|" || valuePart.startsWith("|")) {
				// Optional chomping indicator: |, |+, |-
				result[key] = this.parseBlockLiteral(indent);
			} else if (valuePart === ">" || valuePart.startsWith(">")) {
				result[key] = this.parseFolded(indent);
			} else {
				result[key] = this.parseScalar(valuePart);
			}
		}
		return result;
	}

	private parseList(indent: number): any[] {
		const result: any[] = [];
		while (this.pos < this.lines.length) {
			const line = this.lines[this.pos];
			if (line.indent < indent) break;
			// Deeper-indented lines belong to a previous list item's value.
			if (line.indent > indent) {
				this.pos++;
				continue;
			}
			if (!line.content.startsWith("- ")) break;

			const firstValue = line.content.slice(2).trim();
			this.pos++;

			// Inline mapping on the list item: "- key: value" (or "- key:").
			// Continuation lines at deeper indent are sibling keys of the
			// inline mapping. We parse them inline (without dedenting, to
			// preserve block scalar relative indentation).
			if (firstValue && firstValue.includes(":")) {
				const firstKeyMatch = firstValue.match(/^([^:]+):\s*(.*)$/);
				if (firstKeyMatch && /^[\w-]+$/.test(firstKeyMatch[1].trim())) {
					const obj = this.parseInlineMappingEntry(firstKeyMatch[1].trim(), firstKeyMatch[2], indent);
					result.push(obj);
					continue;
				}
			}

			if (firstValue === "") {
				const next = this.lines[this.pos];
				if (!next || next.indent <= indent) {
					result.push(null);
				} else if (next.content.startsWith("- ")) {
					result.push(this.parseList(next.indent));
				} else {
					result.push(this.parseMapping(next.indent));
				}
			} else {
				result.push(this.parseScalar(firstValue));
			}
		}
		return result;
	}

	/**
	 * Parse a key/value pair within an inline mapping started by a list item.
	 * The first key has been read; we still need to process its value (which
	 * may recurse to consume additional deeper-indented lines), then process
	 * any subsequent sibling keys at deeper indent.
	 *
	 * Unlike parseMapping, we don't expect lines at a single indent — sibling
	 * keys may be at varying indents (e.g., a block scalar's content is at
	 * a deeper indent than the sibling key). The rule: continue consuming
	 * while the current line is at indent > `listIndent` (the parent list's
	 * indent); stop when we hit a line at indent <= listIndent (next list
	 * item, end of list, or parent context).
	 */
	private parseInlineMappingEntry(firstKey: string, firstValuePart: string, listIndent: number): Record<string, any> {
		const obj: Record<string, any> = {};

		// Process the first key
		obj[firstKey] = this.parseInlineValue(firstValuePart, listIndent, this.lines[this.pos]?.indent ?? -1);

		// Process subsequent sibling keys
		while (this.pos < this.lines.length) {
			const line = this.lines[this.pos];
			if (line.indent <= listIndent) break; // next list item or end
			// The line is at indent > listIndent; it's a sibling key (or block scalar content)
			const m = line.content.match(/^([^:]+):\s*(.*)$/);
			if (!m) break; // Not a key — must be content of a previous block scalar (shouldn't happen here)
			const key = m[1].trim();
			const valuePart = m[2];
			this.pos++;
			obj[key] = this.parseInlineValue(valuePart, listIndent, line.indent);
		}
		return obj;
	}

	/**
	 * Parse the value of an inline-mapping key. Handles:
	 *   - explicit scalar
	 *   - explicit block literal (|) or folded (>)
	 *   - empty value: recurse to consume the deeper-indented value
	 */
	private parseInlineValue(valuePart: string, listIndent: number, currentIndent: number): any {
		if (valuePart === "" || valuePart === undefined) {
			// Empty value: recurse into the deeper-indented content.
			const next = this.lines[this.pos];
			if (!next || next.indent <= currentIndent) {
				return null;
			}
			if (next.content.startsWith("- ")) {
				return this.parseList(next.indent);
			}
			return this.parseMapping(next.indent);
		}
		if (valuePart === "|" || valuePart.startsWith("|")) {
			return this.parseBlockLiteral(currentIndent);
		}
		if (valuePart === ">" || valuePart.startsWith(">")) {
			return this.parseFolded(currentIndent);
		}
		return this.parseScalar(valuePart);
	}

	private parseBlockLiteral(parentIndent: number): string {
		// Block indent = indent of the first content line.
		if (this.pos >= this.lines.length) return "";
		const blockIndent = this.lines[this.pos].indent;
		const collected: string[] = [];
		while (this.pos < this.lines.length) {
			const line = this.lines[this.pos];
			if (line.indent < blockIndent) break;
			// Strip exactly `blockIndent` characters from the raw line, preserving
			// any additional indentation beyond the block indent.
			collected.push(line.raw.length > blockIndent ? line.raw.slice(blockIndent) : "");
			this.pos++;
		}
		return collected.join("\n");
	}

	private parseFolded(parentIndent: number): string {
		// Folded: line breaks become spaces (except for blank lines, which
		// become a single newline). We implement the simple case.
		if (this.pos >= this.lines.length) return "";
		const blockIndent = this.lines[this.pos].indent;
		const collected: string[] = [];
		while (this.pos < this.lines.length) {
			const line = this.lines[this.pos];
			if (line.indent < blockIndent) break;
			const content = line.raw.length > blockIndent ? line.raw.slice(blockIndent) : "";
			if (content.trim() === "") {
				collected.push("\n");
			} else {
				collected.push(collected.length > 0 && !collected[collected.length - 1].endsWith("\n") ? " " + content : content);
			}
			this.pos++;
		}
		return collected.join("");
	}

	private parseScalar(s: string): any {
		s = s.trim();
		if (s === "" || s === "~" || s === "null") return null;
		if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
			return s.slice(1, -1);
		}
		if (s === "true" || s === "yes" || s === "on") return true;
		if (s === "false" || s === "no" || s === "off") return false;
		if (/^-?\d+$/.test(s)) return parseInt(s, 10);
		if (/^-?\d+\.\d+$/.test(s)) return parseFloat(s);
		return s;
	}
}

/**
 * Coerce numeric strings to numbers in known numeric fields of the blueprint.
 * YAML often parses bare numbers fine, but when we accept JSON input (e.g.,
 * from loop_design output) numbers are already numbers. This is a no-op for
 * the YAML path; it normalizes JSON path to the same shape.
 */
export function normalizeBlueprint(raw: any): any {
	if (!raw || typeof raw !== "object") return raw;
	const out: any = { ...raw };
	if (typeof out.objective === "string") out.objective = out.objective.trim();
	if (out.stopRules && typeof out.stopRules === "object") {
		const sr = { ...out.stopRules };
		for (const key of ["maxIterations", "maxDurationMs", "noProgressLimit", "verifyTimeoutMs"]) {
			if (typeof sr[key] === "string") sr[key] = parseInt(sr[key], 10);
		}
		// Allow maxDurationMinutes as a synonym for maxDurationMs
		if (typeof sr.maxDurationMinutes === "number" && typeof sr.maxDurationMs !== "number") {
			sr.maxDurationMs = sr.maxDurationMinutes * 60 * 1000;
		}
		out.stopRules = sr;
	}
	if (Array.isArray(out.fanOut)) {
		out.fanOut = out.fanOut.map((step: any) => {
			if (!step || typeof step !== "object") return step;
			const s = { ...step };
			if (typeof s.count === "string") s.count = parseInt(s.count, 10);
			return s;
		});
	}
	return out;
}
