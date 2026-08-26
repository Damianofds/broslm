import {
  createQwen2ByteLevelBpeTokenizer,
  type ByteLevelBpeTokenizer,
  type Qwen2TokenizerParts,
} from "../../../tokenizer";
import {
  readMetadataArray,
  readMetadataNumber,
  type GgufMetadataValue,
} from "./gguf";

export function createQwen2TokenizerFromGgufMetadata(
  metadata: ReadonlyMap<string, GgufMetadataValue>,
): ByteLevelBpeTokenizer {
  return createQwen2ByteLevelBpeTokenizer(createQwen2TokenizerPartsFromGgufMetadata(metadata));
}

export function createQwen2TokenizerPartsFromGgufMetadata(
  metadata: ReadonlyMap<string, GgufMetadataValue>,
): Qwen2TokenizerParts {
  const tokens = readMetadataArray<string>(metadata, "tokenizer.ggml.tokens", "string");
  const merges = readMetadataArray<string>(metadata, "tokenizer.ggml.merges", "string");
  if (!tokens || !merges) {
    throw new Error("Qwen2 GGUF metadata must include tokenizer.ggml.tokens and tokenizer.ggml.merges");
  }

  const tokenTypes = readMetadataArray<number>(metadata, "tokenizer.ggml.token_type", "number");
  return {
    tokens,
    merges,
    tokenTypes: tokenTypes ?? undefined,
    bosTokenId: readMetadataNumber(metadata, "tokenizer.ggml.bos_token_id"),
    eosTokenId: readMetadataNumber(metadata, "tokenizer.ggml.eos_token_id"),
    padTokenId: readMetadataNumber(metadata, "tokenizer.ggml.padding_token_id"),
  };
}
