import { cpSync, createReadStream, existsSync, mkdirSync, statSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import { currentModelExportName } from "./modelName";

const modelsRoot = resolve(__dirname, "../models");
const selectedModelRoot = resolve(modelsRoot, currentModelExportName);

export default defineConfig({
  base: process.env.GITHUB_PAGES === "true" ? "./" : "/",
  server: {
    port: 5173,
    strictPort: false,
  },
  plugins: [serveModelsPlugin(), bundleSelectedModelPlugin()],
});

function bundleSelectedModelPlugin(): Plugin {
  return {
    name: "bundle-selected-model",
    closeBundle() {
      if (!existsSync(selectedModelRoot)) {
        throw new Error(`Selected model export is missing: ${selectedModelRoot}`);
      }

      const outputModelsRoot = resolve(__dirname, "dist/models");
      mkdirSync(outputModelsRoot, { recursive: true });
      cpSync(selectedModelRoot, resolve(outputModelsRoot, currentModelExportName), {
        recursive: true,
        force: true,
      });
    },
  };
}

function serveModelsPlugin(): Plugin {
  return {
    name: "serve-local-models",
    configureServer(server) {
      server.middlewares.use("/models", (request, response, next) => {
        if (!request.url) {
          next();
          return;
        }

        const requestedPath = decodeURIComponent(request.url.split("?")[0] ?? "");
        const filePath = resolve(join(modelsRoot, requestedPath));
        if (!filePath.startsWith(modelsRoot) || !existsSync(filePath)) {
          next();
          return;
        }

        const fileStat = statSync(filePath);
        if (!fileStat.isFile()) {
          next();
          return;
        }

        response.setHeader("Content-Length", String(fileStat.size));
        response.setHeader("Content-Type", contentTypeFor(filePath));
        createReadStream(filePath).pipe(response);
      });
    },
  };
}

function contentTypeFor(filePath: string): string {
  switch (extname(filePath)) {
    case ".json":
      return "application/json; charset=utf-8";
    case ".bin":
      return "application/octet-stream";
    default:
      return "application/octet-stream";
  }
}
