import { describe, expect, it } from "vitest";
import { parseSimpleMarkdown, parseSimpleMarkdownInline } from "./simpleMarkdown";

describe("parseSimpleMarkdown", () => {
  it("parses unordered bullet lists", () => {
    expect(parseSimpleMarkdown("- first\n- **second**")).toEqual({
      hasMarkdownSyntax: true,
      blocks: [
        {
          kind: "unordered-list",
          items: [
            [{ kind: "text", text: "first" }],
            [{ kind: "strong", text: "second" }],
          ],
        },
      ],
    });
  });

  it("parses ordered lists", () => {
    expect(parseSimpleMarkdown("1. first\n2. second")).toEqual({
      hasMarkdownSyntax: true,
      blocks: [
        {
          kind: "ordered-list",
          items: [
            [{ kind: "text", text: "first" }],
            [{ kind: "text", text: "second" }],
          ],
        },
      ],
    });
  });

  it("keeps plain text as paragraphs without markdown syntax", () => {
    expect(parseSimpleMarkdown("Qwen q4_0 output\nplain line")).toEqual({
      hasMarkdownSyntax: false,
      blocks: [
        {
          kind: "paragraph",
          lines: [
            [{ kind: "text", text: "Qwen q4_0 output" }],
            [{ kind: "text", text: "plain line" }],
          ],
        },
      ],
    });
  });
});

describe("parseSimpleMarkdownInline", () => {
  it("parses basic word formatting", () => {
    expect(parseSimpleMarkdownInline("A **bold** and *soft* `token`")).toEqual({
      hasMarkdownSyntax: true,
      inlines: [
        { kind: "text", text: "A " },
        { kind: "strong", text: "bold" },
        { kind: "text", text: " and " },
        { kind: "emphasis", text: "soft" },
        { kind: "text", text: " " },
        { kind: "code", text: "token" },
      ],
    });
  });
});
