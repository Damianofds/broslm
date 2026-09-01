import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const directory = mkdtempSync(join(tmpdir(), "broslm-package-smoke-"));
const packageMetadata = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
);

try {
  execFileSync("npm", ["pack", "--pack-destination", directory], { stdio: "inherit" });
  const tarball = join(directory, `${packageMetadata.name}-${packageMetadata.version}.tgz`);
  execFileSync("npm", ["init", "-y"], { cwd: directory, stdio: "ignore" });
  execFileSync("npm", ["install", tarball, "--ignore-scripts"], {
    cwd: directory,
    stdio: "ignore",
  });
  writeFileSync(
    join(directory, "smoke.mjs"),
    [
      'import { createBroslm, modelOptions } from "broslm";',
      "const client = createBroslm();",
      'const support = await client.checkModelSupport("qwen_cpu_small");',
      'if (!support.supported || modelOptions.length !== 2) process.exit(1);',
      "client.dispose();",
    ].join("\n"),
  );
  execFileSync(process.execPath, [join(directory, "smoke.mjs")], { stdio: "inherit" });

  const installedPackage = JSON.parse(
    readFileSync(join(directory, "node_modules", "broslm", "package.json"), "utf8"),
  );
  if (
    installedPackage.name !== packageMetadata.name ||
    installedPackage.version !== packageMetadata.version
  ) {
    throw new Error("Packed package metadata is incorrect.");
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}
