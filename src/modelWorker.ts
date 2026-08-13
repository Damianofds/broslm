import { installLoaderWorker } from "./engine/src/loader";
import { createModelCacheFetch } from "./modelCache";

installLoaderWorker(undefined, createModelCacheFetch());
