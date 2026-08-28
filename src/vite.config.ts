import { cpSync, createReadStream, existsSync, mkdirSync, statSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

const modelsRoot = resolve(__dirname, "../models");
const optionalModelFolderNames = [
  "qwen2.5-0.5b-instruct-q4_0",
  "qwen2.5-0.5b-instruct-iq1_s",
];

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
      const outputModelsRoot = resolve(__dirname, "dist/models");
      mkdirSync(outputModelsRoot, { recursive: true });
      for (const folderName of optionalModelFolderNames) {
        const optionalModelRoot = resolve(modelsRoot, folderName);
        if (!existsSync(optionalModelRoot)) {
          continue;
        }
        cpSync(optionalModelRoot, resolve(outputModelsRoot, folderName), {
          recursive: true,
          force: true,
        });
      }
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
        if (!filePath.startsWith(modelsRoot)) {
          response.statusCode = 403;
          response.end("Forbidden model asset path");
          return;
        }
        if (!existsSync(filePath)) {
          response.statusCode = 404;
          response.end(`Model asset not found: /models${requestedPath}`);
          return;
        }

        const fileStat = statSync(filePath);
        if (!fileStat.isFile()) {
          response.statusCode = 404;
          response.end(`Model asset not found: /models${requestedPath}`);
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
