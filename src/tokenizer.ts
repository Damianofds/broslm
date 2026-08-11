export interface ByteLevelBpeTokenizer {
  vocabularySize: number;
  eosTokenId: number | null;
  encode(text: string): number[];
  decode(tokenIds: readonly number[]): string;
}

interface TokenizerJson {
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

const TOKEN_PATTERN =
  /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const PAIR_SEPARATOR = "\u0001";

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8", { fatal: false });

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

  const vocabulary: Record<string, number> = vocab;
  const bpeMerges: Array<string | [string, string]> = merges;
  const idToToken = buildIdToToken(vocabulary);
  const mergeRanks = buildMergeRanks(bpeMerges);
  const { byteEncoder, byteDecoder } = buildByteUnicodeMaps();
  const bpeCache = new Map<string, string[]>();
  const eosTokenId = tokenizerJson.added_tokens?.find((token) => token.content === "<|endoftext|>")?.id ?? null;

  function encode(text: string): number[] {
    if (text.length === 0) {
      return [];
    }

    const tokenIds: number[] = [];
    for (const match of text.matchAll(TOKEN_PATTERN)) {
      const piece = match[0] ?? "";
      const byteLevelToken = encodeBytes(piece, byteEncoder);
      const bpeTokens = bytePairEncode(byteLevelToken, mergeRanks, bpeCache);

      for (const token of bpeTokens) {
        const tokenId = vocabulary[token];
        if (tokenId === undefined) {
          throw new Error(`Tokenizer vocab is missing token: ${token}`);
        }
        tokenIds.push(tokenId);
      }
    }

    return tokenIds;
  }

  function decode(tokenIds: readonly number[]): string {
    const bytes: number[] = [];

    for (const tokenId of tokenIds) {
      const token = idToToken[tokenId];
      if (token === undefined) {
        throw new RangeError(`Unknown token id: ${tokenId}`);
      }

      for (const character of Array.from(token)) {
        const byte = byteDecoder.get(character);
        if (byte === undefined) {
          throw new Error(`Tokenizer byte decoder is missing character: ${character}`);
        }
        bytes.push(byte);
      }
    }

    return textDecoder.decode(new Uint8Array(bytes));
  }

  return {
    vocabularySize: idToToken.length,
    eosTokenId,
    encode,
    decode,
  };
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
