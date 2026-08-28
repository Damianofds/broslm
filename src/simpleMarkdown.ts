export interface SimpleMarkdownDocument {
  blocks: SimpleMarkdownBlock[];
  hasMarkdownSyntax: boolean;
}

export type SimpleMarkdownBlock =
  | {
      kind: "paragraph";
      lines: SimpleMarkdownInline[][];
    }
  | {
      kind: "unordered-list";
      items: SimpleMarkdownInline[][];
    }
  | {
      kind: "ordered-list";
      items: SimpleMarkdownInline[][];
    };

export type SimpleMarkdownInline =
  | {
      kind: "text";
      text: string;
    }
  | {
      kind: "strong";
      text: string;
    }
  | {
      kind: "emphasis";
      text: string;
    }
  | {
      kind: "code";
      text: string;
    };

type ListKind = "unordered-list" | "ordered-list";

interface ListLine {
  kind: ListKind;
  text: string;
}

export function parseSimpleMarkdown(source: string): SimpleMarkdownDocument {
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  const blocks: SimpleMarkdownBlock[] = [];
  let hasMarkdownSyntax = false;
  let index = 0;

  while (index < lines.length) {
    const line = lines[index] ?? "";
    if (line.trim().length === 0) {
      index += 1;
      continue;
    }

    const listLine = parseListLine(line);
    if (listLine) {
      const items: SimpleMarkdownInline[][] = [];
      const kind = listLine.kind;
      hasMarkdownSyntax = true;
      while (index < lines.length) {
        const nextListLine = parseListLine(lines[index] ?? "");
        if (!nextListLine || nextListLine.kind !== kind) {
          break;
        }
        const parsedInline = parseSimpleMarkdownInline(nextListLine.text);
        hasMarkdownSyntax ||= parsedInline.hasMarkdownSyntax;
        items.push(parsedInline.inlines);
        index += 1;
      }
      blocks.push({ kind, items });
      continue;
    }

    const paragraphLines: SimpleMarkdownInline[][] = [];
    while (index < lines.length) {
      const nextLine = lines[index] ?? "";
      if (nextLine.trim().length === 0 || parseListLine(nextLine)) {
        break;
      }
      const parsedInline = parseSimpleMarkdownInline(nextLine);
      hasMarkdownSyntax ||= parsedInline.hasMarkdownSyntax;
      paragraphLines.push(parsedInline.inlines);
      index += 1;
    }
    blocks.push({ kind: "paragraph", lines: paragraphLines });
  }

  return { blocks, hasMarkdownSyntax };
}

export function parseSimpleMarkdownInline(source: string): {
  inlines: SimpleMarkdownInline[];
  hasMarkdownSyntax: boolean;
} {
  const inlines: SimpleMarkdownInline[] = [];
  let hasMarkdownSyntax = false;
  let textStart = 0;
  let index = 0;

  while (index < source.length) {
    const marker = markerAt(source, index);
    if (!marker) {
      index += 1;
      continue;
    }

    const closeIndex = source.indexOf(marker.close, index + marker.open.length);
    if (closeIndex < 0 || closeIndex === index + marker.open.length) {
      index += marker.open.length;
      continue;
    }

    if (textStart < index) {
      inlines.push({ kind: "text", text: source.slice(textStart, index) });
    }
    inlines.push({
      kind: marker.kind,
      text: source.slice(index + marker.open.length, closeIndex),
    });
    hasMarkdownSyntax = true;
    index = closeIndex + marker.close.length;
    textStart = index;
  }

  if (textStart < source.length || inlines.length === 0) {
    inlines.push({ kind: "text", text: source.slice(textStart) });
  }

  return { inlines, hasMarkdownSyntax };
}

function parseListLine(line: string): ListLine | null {
  const unorderedMatch = /^\s*[-+*]\s+(.+)$/.exec(line);
  if (unorderedMatch) {
    return { kind: "unordered-list", text: unorderedMatch[1]?.trimEnd() ?? "" };
  }

  const orderedMatch = /^\s*\d+[.)]\s+(.+)$/.exec(line);
  if (orderedMatch) {
    return { kind: "ordered-list", text: orderedMatch[1]?.trimEnd() ?? "" };
  }

  return null;
}

function markerAt(
  source: string,
  index: number,
): { kind: Exclude<SimpleMarkdownInline["kind"], "text">; open: string; close: string } | null {
  if (source.startsWith("**", index)) {
    return { kind: "strong", open: "**", close: "**" };
  }
  if (source.startsWith("__", index)) {
    return { kind: "strong", open: "__", close: "__" };
  }
  if (source[index] === "*") {
    return { kind: "emphasis", open: "*", close: "*" };
  }
  if (source[index] === "`") {
    return { kind: "code", open: "`", close: "`" };
  }
  return null;
}
