export type BroslmErrorCode =
  | "INVALID_STATE"
  | "INVALID_ARGUMENT"
  | "BACKEND_UNAVAILABLE"
  | "MODEL_LOAD_FAILED"
  | "GENERATION_FAILED"
  | "ABORTED"
  | "DISPOSED";

export class BroslmError extends Error {
  readonly code: BroslmErrorCode;

  constructor(code: BroslmErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "BroslmError";
    this.code = code;
  }
}
