export interface Qwen2WebGpuSafetyLimits {
  maxPrefillTokens: number;
  maxSequenceTokens: number;
}

export const qwen2WebGpuSafetyLimits: Qwen2WebGpuSafetyLimits = {
  maxPrefillTokens: 256,
  maxSequenceTokens: 384,
};

export function qwen2WebGpuCacheSequenceLength(modelMaximumSequenceLength: number): number {
  return Math.min(modelMaximumSequenceLength, qwen2WebGpuSafetyLimits.maxSequenceTokens);
}

export function qwen2WebGpuPromptSafetyError(
  promptTokenCount: number,
  requestedNewTokens: number,
): string | null {
  if (promptTokenCount > qwen2WebGpuSafetyLimits.maxPrefillTokens) {
    return (
      `Qwen WebGPU prefill is capped at ${qwen2WebGpuSafetyLimits.maxPrefillTokens} ` +
      `tokens for GPU stability. Current prompt is ${promptTokenCount} tokens.`
    );
  }

  if (promptTokenCount >= qwen2WebGpuSafetyLimits.maxSequenceTokens) {
    return (
      `Qwen WebGPU cache is capped at ${qwen2WebGpuSafetyLimits.maxSequenceTokens} ` +
      `tokens for GPU stability. Current prompt is ${promptTokenCount} tokens.`
    );
  }

  const requestedSequenceLength = promptTokenCount + requestedNewTokens;
  if (requestedSequenceLength > qwen2WebGpuSafetyLimits.maxSequenceTokens) {
    return (
      `Qwen WebGPU will cap this run at ${qwen2WebGpuSafetyLimits.maxSequenceTokens} total ` +
      `tokens; reduce new tokens for the full requested length.`
    );
  }

  return null;
}

export function qwen2WebGpuPrefillSafetyError(promptTokenCount: number): string | null {
  if (promptTokenCount > qwen2WebGpuSafetyLimits.maxPrefillTokens) {
    return (
      `Qwen WebGPU prefill is capped at ${qwen2WebGpuSafetyLimits.maxPrefillTokens} ` +
      `tokens for GPU stability. Current prompt is ${promptTokenCount} tokens.`
    );
  }
  return null;
}
