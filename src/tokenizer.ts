export interface ByteLevelBpeTokenizer {
  vocabularySize: number;
  eosTokenId: number | null;
  encode(text: string): number[];
  decode(tokenIds: readonly number[]): string;
  createIncrementalDecoder(): ByteLevelBpeIncrementalDecoder;
}

export interface ByteLevelBpeIncrementalDecoder {
  push(tokenId: number): string;
  finish(): string;
}

export interface AddedTokenizerToken {
  id: number;
  content: string;
  special?: boolean;
}

export interface TokenizerJson {
  added_tokens?: Array<{
    id: number;
    content: string;
    special?: boolean;
  }>;
  model?: {
    type?: string;
    vocab?: Record<string, number>;
    merges?: Array<string | [string, string]>;
  };
}

export interface ByteLevelBpeTokenizerParts {
  vocab: Record<string, number>;
  merges: Array<string | [string, string]>;
  addedTokens?: readonly AddedTokenizerToken[];
  tokenPattern?: RegExp;
  eosTokenId?: number | null;
}

export interface Qwen2TokenizerParts {
  tokens: readonly string[];
  merges: readonly string[];
  tokenTypes?: readonly number[];
  bosTokenId?: number | null;
  eosTokenId?: number | null;
  padTokenId?: number | null;
}

const TOKEN_PATTERN =
  /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const QWEN2_TOKEN_PATTERN =
  /'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu;
const PAIR_SEPARATOR = "\u0001";
const GGUF_TOKEN_TYPE_CONTROL = 3;
const GGUF_TOKEN_TYPE_USER_DEFINED = 4;

const textEncoder = new TextEncoder();

export async function loadByteLevelBpeTokenizer(
  tokenizerUrl: string,
  fetchImpl: typeof fetch = fetch,
): Promise<ByteLevelBpeTokenizer> {
  const response = await fetchImpl(tokenizerUrl);
  if (!response.ok) {
    throw new Error(`Failed to download tokenizer: HTTP ${response.status}`);
  }

  return createByteLevelBpeTokenizer((await response.json()) as TokenizerJson);
}

export function createByteLevelBpeTokenizer(tokenizerJson: TokenizerJson): ByteLevelBpeTokenizer {
  if (tokenizerJson.model?.type !== "BPE") {
    throw new Error(`Unsupported tokenizer model type: ${tokenizerJson.model?.type ?? "missing"}`);
  }

  const vocab = tokenizerJson.model.vocab;
  const merges = tokenizerJson.model.merges;
  if (!vocab || !Array.isArray(merges)) {
    throw new Error("Tokenizer JSON must contain BPE vocab and merges");
  }

  const eosTokenId =
    tokenizerJson.added_tokens?.find((token) => token.content === "<|endoftext|>")?.id ?? null;

  return createByteLevelBpeTokenizerFromParts({
    vocab,
    merges,
    addedTokens: tokenizerJson.added_tokens,
    eosTokenId,
  });
}

export function createByteLevelBpeTokenizerFromParts(
  parts: ByteLevelBpeTokenizerParts,
): ByteLevelBpeTokenizer {
  const vocabulary: Record<string, number> = { ...parts.vocab };
  for (const token of parts.addedTokens ?? []) {
    vocabulary[token.content] = token.id;
  }

  const bpeMerges: Array<string | [string, string]> = parts.merges;
  const idToToken = buildIdToToken(vocabulary);
  const mergeRanks = buildMergeRanks(bpeMerges);
  const { byteEncoder, byteDecoder } = buildByteUnicodeMaps();
  const bpeCache = new Map<string, string[]>();
  const tokenPattern = parts.tokenPattern ?? TOKEN_PATTERN;
  const specialTokenIds = new Map<string, number>();
  const specialTokenContents = new Set<string>();
  for (const token of parts.addedTokens ?? []) {
    if (token.special) {
      specialTokenIds.set(token.content, token.id);
      specialTokenContents.add(token.content);
    }
  }

  function encode(text: string): number[] {
    if (text.length === 0) {
      return [];
    }

    const tokenIds: number[] = [];
    for (const segment of splitSpecialTokens(text, specialTokenIds)) {
      if (typeof segment === "number") {
        tokenIds.push(segment);
      } else {
        encodeTextSegment(
          segment,
          tokenPattern,
          byteEncoder,
          byteDecoder,
          mergeRanks,
          bpeCache,
          vocabulary,
          tokenIds,
        );
      }
    }

    return tokenIds;
  }

  function tokenContent(tokenId: number): { special?: string; bytes?: Uint8Array } {
      const token = idToToken[tokenId];
      if (token === undefined) {
        throw new RangeError(`Unknown token id: ${tokenId}`);
      }

      if (specialTokenContents.has(token)) {
        return { special: token };
      }

      const fallbackByte = parseByteFallbackToken(token);
      if (fallbackByte !== null) {
        return { bytes: Uint8Array.of(fallbackByte) };
      }

      const bytes: number[] = [];
      for (const character of Array.from(token)) {
        const byte = byteDecoder.get(character);
        if (byte === undefined) {
          throw new Error(`Tokenizer byte decoder is missing character: ${character}`);
        }
        bytes.push(byte);
      }
      return { bytes: Uint8Array.from(bytes) };
  }

  function createIncrementalDecoder(): ByteLevelBpeIncrementalDecoder {
    const decoder = new TextDecoder("utf-8", { fatal: false });
    return {
      push(tokenId: number): string {
        const content = tokenContent(tokenId);
        if (content.special !== undefined) {
          return decoder.decode() + content.special;
        }
        return decoder.decode(content.bytes, { stream: true });
      },
      finish(): string {
        return decoder.decode();
      },
    };
  }

  function decode(tokenIds: readonly number[]): string {
    const decoder = createIncrementalDecoder();
    let decoded = "";
    for (const tokenId of tokenIds) {
      decoded += decoder.push(tokenId);
    }
    return decoded + decoder.finish();
  }

  return {
    vocabularySize: idToToken.length,
    eosTokenId: parts.eosTokenId ?? null,
    encode,
    decode,
    createIncrementalDecoder,
  };
}

export function createQwen2ByteLevelBpeTokenizer(
  parts: Qwen2TokenizerParts,
): ByteLevelBpeTokenizer {
  const vocab: Record<string, number> = {};
  const addedTokens: AddedTokenizerToken[] = [];
  for (let id = 0; id < parts.tokens.length; id += 1) {
    const content = parts.tokens[id];
    if (content === undefined) {
      continue;
    }
    vocab[content] = id;
    if (isQwen2SpecialToken(content, parts.tokenTypes?.[id])) {
      addedTokens.push({
        id,
        content,
        special: true,
      });
    }
  }

  return createByteLevelBpeTokenizerFromParts({
    vocab,
    merges: [...parts.merges],
    addedTokens,
    tokenPattern: QWEN2_TOKEN_PATTERN,
    eosTokenId: parts.eosTokenId ?? null,
  });
}

function encodeTextSegment(
  text: string,
  tokenPattern: RegExp,
  byteEncoder: Map<number, string>,
  byteDecoder: Map<string, number>,
  mergeRanks: ReadonlyMap<string, number>,
  bpeCache: Map<string, string[]>,
  vocabulary: Record<string, number>,
  tokenIds: number[],
): void {
  if (text.length === 0) {
    return;
  }

  tokenPattern.lastIndex = 0;
  for (const match of text.matchAll(tokenPattern)) {
    const piece = match[0] ?? "";
    const byteLevelToken = encodeBytes(piece, byteEncoder);
    const bpeTokens = bytePairEncode(byteLevelToken, mergeRanks, bpeCache);

    for (const token of bpeTokens) {
      const tokenId = vocabulary[token];
      if (tokenId === undefined) {
        const fallbackTokenIds = encodeByteFallbackTokenIds(token, byteDecoder, vocabulary);
        if (!fallbackTokenIds) {
          throw new Error(`Tokenizer vocab is missing token: ${token}`);
        }
        tokenIds.push(...fallbackTokenIds);
        continue;
      }
      tokenIds.push(tokenId);
    }
  }
}

function splitSpecialTokens(
  text: string,
  specialTokenIds: ReadonlyMap<string, number>,
): Array<string | number> {
  if (specialTokenIds.size === 0) {
    return [text];
  }

  const specialTokens = [...specialTokenIds.keys()].sort((left, right) => right.length - left.length);
  const segments: Array<string | number> = [];
  let cursor = 0;
  while (cursor < text.length) {
    let matchedToken: string | null = null;
    for (const token of specialTokens) {
      if (text.startsWith(token, cursor)) {
        matchedToken = token;
        break;
      }
    }

    if (matchedToken) {
      segments.push(specialTokenIds.get(matchedToken) ?? 0);
      cursor += matchedToken.length;
      continue;
    }

    let nextSpecialIndex = text.length;
    for (const token of specialTokens) {
      const index = text.indexOf(token, cursor + 1);
      if (index >= 0 && index < nextSpecialIndex) {
        nextSpecialIndex = index;
      }
    }
    segments.push(text.slice(cursor, nextSpecialIndex));
    cursor = nextSpecialIndex;
  }

  return segments;
}

function buildIdToToken(vocab: Record<string, number>): string[] {
  const idToToken: string[] = [];
  for (const [token, tokenId] of Object.entries(vocab)) {
    if (!Number.isInteger(tokenId) || tokenId < 0) {
      throw new Error(`Invalid tokenizer id for token ${token}: ${tokenId}`);
    }
    idToToken[tokenId] = token;
  }

  return idToToken;
}

function encodeByteFallbackTokenIds(
  token: string,
  byteDecoder: ReadonlyMap<string, number>,
  vocabulary: Record<string, number>,
): number[] | null {
  const tokenIds: number[] = [];
  for (const character of Array.from(token)) {
    const byte = byteDecoder.get(character);
    if (byte === undefined) {
      return null;
    }

    const tokenId = vocabulary[formatByteFallbackToken(byte)];
    if (tokenId === undefined) {
      return null;
    }
    tokenIds.push(tokenId);
  }
  return tokenIds;
}

function formatByteFallbackToken(byte: number): string {
  return `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
}

function parseByteFallbackToken(token: string): number | null {
  const match = /^<0x([0-9A-Fa-f]{2})>$/.exec(token);
  if (!match) {
    return null;
  }
  return Number.parseInt(match[1] ?? "0", 16);
}

function isQwen2SpecialToken(content: string, tokenType: number | undefined): boolean {
  return (
    tokenType === GGUF_TOKEN_TYPE_CONTROL ||
    tokenType === GGUF_TOKEN_TYPE_USER_DEFINED ||
    (content.startsWith("<|") && content.endsWith("|>"))
  );
}

function buildMergeRanks(merges: Array<string | [string, string]>): Map<string, number> {
  const ranks = new Map<string, number>();
  for (let rank = 0; rank < merges.length; rank += 1) {
    const merge = merges[rank];
    const pair = Array.isArray(merge) ? merge : splitMerge(merge);
    ranks.set(pairKey(pair[0], pair[1]), rank);
  }
  return ranks;
}

function splitMerge(merge: string): [string, string] {
  const separatorIndex = merge.indexOf(" ");
  if (separatorIndex < 0) {
    throw new Error(`Invalid BPE merge entry: ${merge}`);
  }
  return [merge.slice(0, separatorIndex), merge.slice(separatorIndex + 1)];
}

function buildByteUnicodeMaps(): {
  byteEncoder: Map<number, string>;
  byteDecoder: Map<string, number>;
} {
  const bytes: number[] = [];
  for (let byte = 33; byte <= 126; byte += 1) {
    bytes.push(byte);
  }
  for (let byte = 161; byte <= 172; byte += 1) {
    bytes.push(byte);
  }
  for (let byte = 174; byte <= 255; byte += 1) {
    bytes.push(byte);
  }

  const codePoints = [...bytes];
  let nextCodePointOffset = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    if (!bytes.includes(byte)) {
      bytes.push(byte);
      codePoints.push(256 + nextCodePointOffset);
      nextCodePointOffset += 1;
    }
  }

  const byteEncoder = new Map<number, string>();
  const byteDecoder = new Map<string, number>();
  for (let index = 0; index < bytes.length; index += 1) {
    const byte = bytes[index] ?? 0;
    const character = String.fromCodePoint(codePoints[index] ?? 0);
    byteEncoder.set(byte, character);
    byteDecoder.set(character, byte);
  }

  return { byteEncoder, byteDecoder };
}

function encodeBytes(text: string, byteEncoder: Map<number, string>): string {
  let encoded = "";
  for (const byte of textEncoder.encode(text)) {
    const character = byteEncoder.get(byte);
    if (character === undefined) {
      throw new Error(`Tokenizer byte encoder is missing byte: ${byte}`);
    }
    encoded += character;
  }
  return encoded;
}

function bytePairEncode(
  token: string,
  mergeRanks: ReadonlyMap<string, number>,
  cache: Map<string, string[]>,
): string[] {
  const cached = cache.get(token);
  if (cached) {
    return cached;
  }

  let parts = Array.from(token);
  if (parts.length <= 1) {
    cache.set(token, parts);
    return parts;
  }

  while (parts.length > 1) {
    const bestPair = findBestMergePair(parts, mergeRanks);
    if (!bestPair) {
      break;
    }

    const nextParts: string[] = [];
    for (let index = 0; index < parts.length; index += 1) {
      if (
        index < parts.length - 1 &&
        parts[index] === bestPair.left &&
        parts[index + 1] === bestPair.right
      ) {
        nextParts.push(bestPair.left + bestPair.right);
        index += 1;
      } else {
        nextParts.push(parts[index] ?? "");
      }
    }
    parts = nextParts;
  }

  cache.set(token, parts);
  return parts;
}

function findBestMergePair(
  parts: readonly string[],
  mergeRanks: ReadonlyMap<string, number>,
): { left: string; right: string } | null {
  let bestRank = Number.POSITIVE_INFINITY;
  let bestPair: { left: string; right: string } | null = null;

  for (let index = 0; index < parts.length - 1; index += 1) {
    const left = parts[index] ?? "";
    const right = parts[index + 1] ?? "";
    const rank = mergeRanks.get(pairKey(left, right));
    if (rank !== undefined && rank < bestRank) {
      bestRank = rank;
      bestPair = { left, right };
    }
  }

  return bestPair;
}

function pairKey(left: string, right: string): string {
  return `${left}${PAIR_SEPARATOR}${right}`;
}
