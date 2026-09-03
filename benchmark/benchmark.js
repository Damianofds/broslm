import { createBroslm } from "../dist/browser.js";

const runButton = document.querySelector("#run");
const output = document.querySelector("#output");
const cachedLengths = [256, 1_024, 8_192, 16_384, 24_576, 32_752];

runButton.addEventListener("click", () => void runBenchmark());

async function runBenchmark() {
  runButton.disabled = true;
  output.textContent = "";
  const write = (value) => {
    output.textContent += `${JSON.stringify(value, null, 2)}\n`;
  };

  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter is available.");
    write({
      kind: "adapter",
      info: adapter.info ?? null,
      features: [...adapter.features].sort(),
      limits: Object.fromEntries(
        [
          "maxBufferSize",
          "maxStorageBufferBindingSize",
          "maxComputeWorkgroupStorageSize",
          "maxComputeInvocationsPerWorkgroup",
        ].map((key) => [key, adapter.limits[key]]),
      ),
    });

    const client = createBroslm({ logLevel: "warn" });
    const support = await client.checkModelSupport("qwen");
    if (!support.supported) throw new Error(support.reason ?? "Qwen model is unsupported.");
    await client.loadModel("qwen");

    for (const targetLength of cachedLengths) {
      const prompt = promptNearTokenCount(client, targetLength);
      const inputTokens = client.countPromptTokens(prompt);
      const before = client.diagnostics.runtime;
      const tokenLatencies = [];
      let firstTokenMs = null;
      const startedAt = performance.now();
      let previousAt = startedAt;
      let generatedTokens = 0;

      for await (const chunk of client.stream(prompt, { maxTokens: 16, temperature: 0, topK: 1 })) {
        const now = performance.now();
        firstTokenMs ??= now - startedAt;
        tokenLatencies.push(now - previousAt);
        previousAt = now;
        generatedTokens += 1;
        void chunk;
      }

      const elapsedMs = performance.now() - startedAt;
      const after = client.diagnostics.runtime;
      const decodeElapsedMs = Math.max(0, elapsedMs - (firstTokenMs ?? elapsedMs));
      const decodedTokens = Math.max(0, generatedTokens - 1);
      const runtimeDelta = subtractDiagnostics(after, before);
      write({
        kind: "run",
        targetLength,
        inputTokens,
        generatedTokens,
        timeToFirstTokenMs: firstTokenMs,
        prefillTokensPerSecond: firstTokenMs ? inputTokens / (firstTokenMs / 1_000) : null,
        elapsedMs,
        outputTokensPerSecond: decodeElapsedMs > 0 ? decodedTokens / (decodeElapsedMs / 1_000) : null,
        tokenLatencyMs: percentileSummary(tokenLatencies.slice(1)),
        buffersCreatedPerToken: perToken(runtimeDelta?.buffersCreated, generatedTokens),
        bytesReadBackPerToken: perToken(runtimeDelta?.bytesReadBack, generatedTokens),
        runtimeDelta,
      });
    }
    client.dispose();
  } catch (error) {
    write({ kind: "error", message: error instanceof Error ? error.message : String(error) });
  } finally {
    runButton.disabled = false;
  }
}

function perToken(value, tokenCount) {
  return typeof value === "number" && tokenCount > 0 ? value / tokenCount : null;
}

function promptNearTokenCount(client, targetTokens) {
  const unit = " WebGPU long context benchmark token.";
  let low = 1;
  let high = 2;
  while (client.countPromptTokens(unit.repeat(high)) < targetTokens) high *= 2;
  while (low + 1 < high) {
    const middle = Math.floor((low + high) / 2);
    if (client.countPromptTokens(unit.repeat(middle)) < targetTokens) low = middle;
    else high = middle;
  }
  const candidates = [low, high].map((count) => unit.repeat(count));
  return candidates.sort(
    (left, right) =>
      Math.abs(client.countPromptTokens(left) - targetTokens) -
      Math.abs(client.countPromptTokens(right) - targetTokens),
  )[0];
}

function percentileSummary(values) {
  if (values.length === 0) return { p50: null, p95: null };
  const sorted = [...values].sort((left, right) => left - right);
  return {
    p50: sorted[Math.floor((sorted.length - 1) * 0.5)],
    p95: sorted[Math.floor((sorted.length - 1) * 0.95)],
  };
}

function subtractDiagnostics(after, before) {
  if (!after || !before) return null;
  return Object.fromEntries(Object.keys(after).map((key) => [key, after[key] - before[key]]));
}
