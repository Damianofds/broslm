import { afterEach, describe, expect, it, vi } from "vitest";
import { createBroslmLogger } from "../src/logger";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("broSLM logger", () => {
  it("uses WARN as its default threshold", () => {
    const debug = vi.spyOn(console, "debug").mockImplementation(() => undefined);
    const info = vi.spyOn(console, "info").mockImplementation(() => undefined);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const logger = createBroslmLogger();

    logger.debug("debug-message");
    logger.info("info-message");
    logger.warn("warn-message", { value: 1 });
    logger.error("error-message");

    expect(debug).not.toHaveBeenCalled();
    expect(info).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalledWith("[broslm] warn-message", { value: 1 });
    expect(error).toHaveBeenCalledWith("[broslm] error-message");
  });

  it("enables every level at DEBUG and prefixes every message", () => {
    const spies = {
      debug: vi.spyOn(console, "debug").mockImplementation(() => undefined),
      info: vi.spyOn(console, "info").mockImplementation(() => undefined),
      warn: vi.spyOn(console, "warn").mockImplementation(() => undefined),
      error: vi.spyOn(console, "error").mockImplementation(() => undefined),
    };
    const logger = createBroslmLogger("debug");

    for (const level of ["debug", "info", "warn", "error"] as const) {
      logger[level](`${level}-message`);
      expect(spies[level]).toHaveBeenCalledWith(`[broslm] ${level}-message`);
    }
  });
});
