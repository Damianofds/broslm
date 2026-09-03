import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const source = readFileSync(new URL("../src/qwen2/gpuModel.ts", import.meta.url), "utf8");
const shaderMatch = source.match(
  /const\s+qwen2QuantizedMatrixVectorCooperativeShader\s*=\s*`([\s\S]*?)`;/,
);
if (!shaderMatch) {
  throw new Error("The cooperative quantized GEMV shader was not found");
}

const html = `<!doctype html><body>pending<script>
(async () => {
  try {
  const shaderCode = ${JSON.stringify(shaderMatch[1])};
  const inputSize = 32;
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter is available");
  const device = await adapter.requestDevice();
  const rowsPerDispatch = device.limits.maxComputeWorkgroupsPerDimension;
  const outputSize = rowsPerDispatch + 2;
  const input = Float32Array.from({ length: inputSize }, (_, index) => (index - 15.5) / 16);

  async function runCase(quantType) {
    const rowByteLength = quantType === 4 ? 20 : 36;
    const weight = new Uint8Array(outputSize * rowByteLength);
    const bias = new Float32Array(outputSize);
    const expected = new Float32Array(outputSize);

    for (let row = 0; row < outputSize; row += 1) {
      const rowOffset = row * rowByteLength;
      weight[rowOffset] = 0;
      weight[rowOffset + 1] = 0x38;
      bias[row] = ((row % 17) - 8) / 8;
      let sum = bias[row];
      if (quantType === 4) {
        for (let index = 0; index < 16; index += 1) {
          const low = (row + index * 3) % 16;
          const high = (row + (index + 16) * 3) % 16;
          weight[rowOffset + 4 + index] = low | (high << 4);
          sum += (low - 8) * 0.5 * input[index];
          sum += (high - 8) * 0.5 * input[index + 16];
        }
      } else {
        for (let index = 0; index < inputSize; index += 1) {
          const quantized = (row * 7 + index * 5) % 255 - 127;
          weight[rowOffset + 4 + index] = quantized & 0xff;
          sum += quantized * 0.5 * input[index];
        }
      }
      expected[row] = sum;
    }

    const createBuffer = (size, usage, data) => {
      const buffer = device.createBuffer({ size, usage });
      if (data) device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const weightBuffer = createBuffer(
      weight.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      weight,
    );
    const inputBuffer = createBuffer(
      input.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      input,
    );
    const biasBuffer = createBuffer(
      bias.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      bias,
    );
    const params = new Uint32Array([
      inputSize,
      outputSize,
      1,
      0,
      rowByteLength,
      quantType,
      1,
      0,
      rowsPerDispatch,
    ]);
    const paramsBuffer = createBuffer(
      params.byteLength,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      params,
    );
    const outputBuffer = createBuffer(
      expected.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const readback = createBuffer(
      expected.byteLength,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );

    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code: shaderCode });
    const compilation = await module.getCompilationInfo();
    const compilationErrors = compilation.messages.filter((message) => message.type === "error");
    if (compilationErrors.length > 0) {
      throw new Error(compilationErrors.map((message) => message.message).join("; "));
    }
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: weightBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: biasBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(rowsPerDispatch, Math.ceil(outputSize / rowsPerDispatch));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, expected.byteLength);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    if (validationError) throw validationError;

    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    let maximumError = 0;
    for (let row = 0; row < outputSize; row += 1) {
      const error = Math.abs(actual[row] - expected[row]);
      const tolerance = Math.max(0.001, Math.abs(expected[row]) * 0.00001);
      if (!Number.isFinite(actual[row]) || error > tolerance) {
        throw new Error(
          "Q" + quantType + " row " + row + " expected " + expected[row] +
            ", received " + actual[row] + " (error " + error + ")",
        );
      }
      maximumError = Math.max(maximumError, error);
    }

    for (const buffer of [
      weightBuffer,
      inputBuffer,
      biasBuffer,
      paramsBuffer,
      outputBuffer,
      readback,
    ]) buffer.destroy();
    return {
      quantType,
      maximumError,
      boundaryRows: Array.from(actual.slice(rowsPerDispatch - 1, rowsPerDispatch + 2)),
    };
  }

    const cases = [await runCase(4), await runCase(8)];
    document.body.textContent = JSON.stringify({
      ok: true,
      outputSize,
      dispatch: [rowsPerDispatch, Math.ceil(outputSize / rowsPerDispatch)],
      cases,
    });
  } catch (error) {
    document.body.textContent = JSON.stringify({
      ok: false,
      error: error?.message ?? String(error),
    });
  }
})();
</script>`;

const chrome = process.env.BROSLM_CHROME_PATH ?? "google-chrome";
const result = spawnSync(
  chrome,
  [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu-sandbox",
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--use-angle=swiftshader",
    "--virtual-time-budget=120000",
    "--dump-dom",
    `data:text/html;charset=utf-8,${encodeURIComponent(html)}`,
  ],
  { encoding: "utf8", maxBuffer: 4 * 1024 * 1024, timeout: 180000 },
);

if (result.error) throw result.error;
const output = result.stdout.trim();
console.log(output);
if (result.status !== 0 || (!output.includes('"ok":true') && !output.includes("&quot;ok&quot;:true"))) {
  if (result.stderr) console.error(result.stderr.trim());
  process.exitCode = 1;
}
