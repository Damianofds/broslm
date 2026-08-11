import { currentModelExportName } from "./modelName";

const publicBaseUrl = import.meta.env.BASE_URL;

export { currentModelExportName };
export const modelBaseUrl = `${publicBaseUrl}models/${currentModelExportName}/`;
export const tokenizerUrl = `${publicBaseUrl}tokenizer/tinystories-tokenizer.json`;
