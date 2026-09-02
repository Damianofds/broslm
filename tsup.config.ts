import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    browser: "src/browser.ts",
  },
  format: ["esm"],
  target: "es2022",
  dts: false,
  splitting: false,
  sourcemap: true,
  clean: true,
  treeshake: true,
});
