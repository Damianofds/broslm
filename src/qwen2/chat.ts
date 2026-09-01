import { BroslmError } from "../errors";
import type { ChatMessage, ChatRole, PromptInput } from "../types";

const DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant.";
const CHAT_ROLES = new Set<ChatRole>(["system", "user", "assistant"]);

export function formatQwen2Prompt(input: PromptInput): string {
  const messages = normalizePrompt(input);
  validateMessages(messages);

  const conversation = messages[0]?.role === "system"
    ? messages
    : [{ role: "system", content: DEFAULT_SYSTEM_MESSAGE } as const, ...messages];

  return (
    conversation
      .map(({ role, content }) => `<|im_start|>${role}\n${content}<|im_end|>\n`)
      .join("") + "<|im_start|>assistant\n"
  );
}

function normalizePrompt(input: PromptInput): readonly ChatMessage[] {
  if (typeof input === "string") {
    return [{
      role: "user",
      content: input.replace(/\r\n?/g, "\n").replace(/\n+/g, "\n"),
    }];
  }
  if (!Array.isArray(input)) {
    throw invalidPrompt("Prompt input must be a string or an array of chat messages.");
  }
  return input;
}

function validateMessages(messages: readonly ChatMessage[]): void {
  if (messages.length === 0) {
    throw invalidPrompt("Prompt messages must not be empty.");
  }

  let systemMessages = 0;
  let hasUserMessage = false;
  for (const [index, message] of messages.entries()) {
    if (!message || typeof message !== "object") {
      throw invalidPrompt(`Prompt message ${index} must be an object.`);
    }
    if (!CHAT_ROLES.has(message.role)) {
      throw invalidPrompt(`Prompt message ${index} has unsupported role: ${String(message.role)}.`);
    }
    if (typeof message.content !== "string" || message.content.length === 0) {
      throw invalidPrompt(`Prompt message ${index} must have non-empty string content.`);
    }
    if (message.role === "system") {
      systemMessages += 1;
      if (index !== 0 || systemMessages > 1) {
        throw invalidPrompt("A single system message is allowed only at the start of the prompt.");
      }
    }
    if (message.role === "user") {
      hasUserMessage = true;
    }
  }

  if (!hasUserMessage) {
    throw invalidPrompt("Prompt messages must include at least one user message.");
  }
  if (messages.at(-1)?.role !== "user") {
    throw invalidPrompt("The final prompt message must have the user role.");
  }
}

function invalidPrompt(message: string): BroslmError {
  return new BroslmError("INVALID_ARGUMENT", message);
}
