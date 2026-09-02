import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
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
  const installedRoot = join(directory, "node_modules", "broslm");
  const installedPackage = JSON.parse(readFileSync(join(installedRoot, "package.json"), "utf8"));
  if (
    installedPackage.name !== packageMetadata.name ||
    installedPackage.version !== packageMetadata.version
  ) {
    throw new Error("Packed package metadata is incorrect.");
  }
  if (installedPackage.exports?.["."]?.node || existsSync(join(installedRoot, "dist", "node.js"))) {
    throw new Error("Packed package must not expose a Node runtime.");
  }
  if (
    !existsSync(join(installedRoot, "dist", "browser.js")) ||
    !existsSync(join(installedRoot, "dist", "browser.d.ts"))
  ) {
    throw new Error("Packed package is missing its browser runtime.");
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}
