import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const source = readFileSync(new URL("../src/qwen2/gpuModel.ts", import.meta.url), "utf8");
const shaders = [...source.matchAll(/const\s+(\w+Shader)\s*=\s*`([\s\S]*?)`;/g)].map(
  ([, name, code]) => ({ name, code }),
);

if (shaders.length === 0) {
  throw new Error("No WGSL shaders were found in src/qwen2/gpuModel.ts");
}

const html = `<!doctype html><body>pending<script>
(async () => {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter is available");
    const device = await adapter.requestDevice();
    const failures = [];
    for (const shader of ${JSON.stringify(shaders)}) {
      const module = device.createShaderModule({ code: shader.code, label: shader.name });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((message) => message.type === "error");
      if (errors.length > 0) failures.push({ name: shader.name, errors });
    }
    document.body.textContent = JSON.stringify({ ok: failures.length === 0, shaderCount: ${shaders.length}, failures });
  } catch (error) {
    document.body.textContent = JSON.stringify({ ok: false, error: error?.message ?? String(error) });
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
    "--virtual-time-budget=15000",
    "--dump-dom",
    `data:text/html;charset=utf-8,${encodeURIComponent(html)}`,
  ],
  { encoding: "utf8", maxBuffer: 4 * 1024 * 1024 },
);

if (result.error) throw result.error;
const output = result.stdout.trim();
console.log(output);
if (result.status !== 0 || (!output.includes('"ok":true') && !output.includes("&quot;ok&quot;:true"))) {
  if (result.stderr) console.error(result.stderr.trim());
  process.exitCode = 1;
}
