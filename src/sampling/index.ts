export interface SamplingOptions {
  temperature?: number;
  topK?: number;
  random?: () => number;
}

export function sampleTokenFromLogits(
  logits: Float32Array,
  options: SamplingOptions = {},
): number {
  if (logits.length === 0) {
    throw new Error("sampleTokenFromLogits requires at least one logit");
  }

  const temperature = options.temperature ?? 0;
  const topK = clampTopK(options.topK ?? 1, logits.length);
  if (!Number.isFinite(temperature) || temperature <= 0 || topK === 1) {
    return argmax(logits);
  }

  const candidates = topKLogits(logits, topK);
  const scale = 1 / temperature;
  let maxScaledLogit = Number.NEGATIVE_INFINITY;
  for (const candidate of candidates) {
    maxScaledLogit = Math.max(maxScaledLogit, candidate.logit * scale);
  }

  let totalWeight = 0;
  const weights = new Float64Array(candidates.length);
  for (let index = 0; index < candidates.length; index += 1) {
    const candidate = candidates[index];
    const weight = Math.exp(((candidate?.logit ?? 0) * scale) - maxScaledLogit);
    weights[index] = weight;
    totalWeight += weight;
  }

  if (!Number.isFinite(totalWeight) || totalWeight <= 0) {
    return candidates[0]?.tokenId ?? argmax(logits);
  }

  const random = options.random ?? Math.random;
  const sample = Math.max(0, Math.min(0.999999999999, random())) * totalWeight;
  let cumulative = 0;
  for (let index = 0; index < candidates.length; index += 1) {
    cumulative += weights[index] ?? 0;
    if (sample < cumulative) {
      return candidates[index]?.tokenId ?? 0;
    }
  }

  return candidates[candidates.length - 1]?.tokenId ?? 0;
}

export function sampleTokenFromCandidates(
  tokenIds: Uint32Array,
  logits: Float32Array,
  options: SamplingOptions = {},
): number {
  if (tokenIds.length === 0 || tokenIds.length !== logits.length) {
    throw new Error("sampleTokenFromCandidates requires matching non-empty candidate arrays");
  }
  const temperature = options.temperature ?? 0;
  if (!Number.isFinite(temperature) || temperature <= 0 || tokenIds.length === 1) {
    return tokenIds[0] ?? 0;
  }

  const scale = 1 / temperature;
  const maximum = logits[0] ?? 0;
  const weights = new Float64Array(logits.length);
  let totalWeight = 0;
  for (let index = 0; index < logits.length; index += 1) {
    const weight = Math.exp(((logits[index] ?? 0) - maximum) * scale);
    weights[index] = weight;
    totalWeight += weight;
  }
  if (!Number.isFinite(totalWeight) || totalWeight <= 0) {
    return tokenIds[0] ?? 0;
  }

  const random = options.random ?? Math.random;
  const sample = Math.max(0, Math.min(0.999999999999, random())) * totalWeight;
  let cumulative = 0;
  for (let index = 0; index < weights.length; index += 1) {
    cumulative += weights[index] ?? 0;
    if (sample < cumulative) {
      return tokenIds[index] ?? tokenIds[0] ?? 0;
    }
  }
  return tokenIds[tokenIds.length - 1] ?? 0;
}

function clampTopK(topK: number, vocabularySize: number): number {
  if (!Number.isFinite(topK)) {
    return 1;
  }
  return Math.max(1, Math.min(vocabularySize, Math.round(topK)));
}

function argmax(values: Float32Array): number {
  if (values.length === 0) {
    throw new Error("argmax requires at least one value");
  }
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (
      (values[index] ?? Number.NEGATIVE_INFINITY) >
      (values[bestIndex] ?? Number.NEGATIVE_INFINITY)
    ) {
      bestIndex = index;
    }
  }
  return bestIndex;
}

function topKLogits(logits: Float32Array, topK: number): Array<{ tokenId: number; logit: number }> {
  const candidates: Array<{ tokenId: number; logit: number }> = [];

  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    const logit = logits[tokenId] ?? Number.NEGATIVE_INFINITY;
    if (!Number.isFinite(logit)) {
      continue;
    }

    if (candidates.length === 0) {
      candidates.push({ tokenId, logit });
      continue;
    }

    if (candidates.length === topK && logit <= (candidates[candidates.length - 1]?.logit ?? 0)) {
      continue;
    }

    let insertAt = candidates.length;
    while (insertAt > 0 && logit > (candidates[insertAt - 1]?.logit ?? Number.NEGATIVE_INFINITY)) {
      insertAt -= 1;
    }
    candidates.splice(insertAt, 0, { tokenId, logit });

    if (candidates.length > topK) {
      candidates.pop();
    }
  }

  if (candidates.length === 0) {
    const tokenId = argmax(logits);
    return [{ tokenId, logit: logits[tokenId] ?? 0 }];
  }

  return candidates;
}
