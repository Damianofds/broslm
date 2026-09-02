export type BroslmLogLevel = "debug" | "info" | "warn" | "error";

export interface BroslmLogger {
  debug(message: string, context?: unknown): void;
  info(message: string, context?: unknown): void;
  warn(message: string, context?: unknown): void;
  error(message: string, context?: unknown): void;
}

const priorities: Readonly<Record<BroslmLogLevel, number>> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

export function createBroslmLogger(level: BroslmLogLevel = "warn"): BroslmLogger {
  const write = (messageLevel: BroslmLogLevel, message: string, context?: unknown): void => {
    if (priorities[messageLevel] < priorities[level]) {
      return;
    }

    const prefixedMessage = `[broslm] ${message}`;
    if (context === undefined) {
      console[messageLevel](prefixedMessage);
    } else {
      console[messageLevel](prefixedMessage, context);
    }
  };

  return {
    debug: (message, context) => write("debug", message, context),
    info: (message, context) => write("info", message, context),
    warn: (message, context) => write("warn", message, context),
    error: (message, context) => write("error", message, context),
  };
}
