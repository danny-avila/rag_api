import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { open, type Entry, type ZipFile } from "yauzl";
import { z } from "zod";
import type { ExtractionResult } from "./contract";
import { ExtractionError } from "./contract";

const requestSchema = z.object({
  path: z.string(),
  maxOutputBytes: z.number().int().positive(),
  maxEntryBytes: z.number().int().positive(),
  maxArchiveBytes: z.number().int().positive(),
  maxEntries: z.number().int().positive(),
});
export type WorkerRequest = z.infer<typeof requestSchema>;
const IMAGE =
  /\.(?:jpe?g|png|gif|tiff?|bmp|webp|jp2|jpx|avif|heic|heif|emf|wmf|svg)$/i;
const PREVIEW = /^(?:docProps|Thumbnails)\//i;

function inspectArchive(path: string, limits: WorkerRequest): Promise<boolean> {
  return new Promise((resolve, reject) => {
    open(
      path,
      { lazyEntries: true, validateEntrySizes: true },
      (error, zip) => {
        if (error || !zip) {
          reject(new ExtractionError("ARCHIVE_INVALID"));
          return;
        }
        let settled = false;
        let total = 0;
        let media = false;
        const names = new Set<string>();
        const fail = (code: "ARCHIVE_INVALID" | "ZIP_BOMB") => {
          if (settled) return;
          settled = true;
          zip.close();
          reject(new ExtractionError(code));
        };
        zip.on("error", () => fail("ARCHIVE_INVALID"));
        zip.on("end", () => {
          if (settled) return;
          if (
            !names.has("[Content_Types].xml") ||
            !names.has("word/document.xml")
          ) {
            fail("ARCHIVE_INVALID");
            return;
          }
          settled = true;
          resolve(media);
        });
        if (zip.entryCount > limits.maxEntries) {
          fail("ZIP_BOMB");
          return;
        }
        zip.on("entry", (entry: Entry) => {
          if (names.has(entry.fileName)) {
            fail("ARCHIVE_INVALID");
            return;
          }
          names.add(entry.fileName);
          if (/\/$/.test(entry.fileName)) {
            zip.readEntry();
            return;
          }
          if (
            entry.uncompressedSize > limits.maxEntryBytes ||
            total + entry.uncompressedSize > limits.maxArchiveBytes
          ) {
            fail("ZIP_BOMB");
            return;
          }
          if (
            (IMAGE.test(entry.fileName) && !PREVIEW.test(entry.fileName)) ||
            /^word\/embeddings\//i.test(entry.fileName)
          )
            media = true;
          zip.openReadStream(entry, (streamError, stream) => {
            if (streamError || !stream) {
              fail("ARCHIVE_INVALID");
              return;
            }
            let bytes = 0;
            stream.on("data", (chunk: Buffer) => {
              bytes += chunk.byteLength;
              total += chunk.byteLength;
              if (
                bytes > limits.maxEntryBytes ||
                total > limits.maxArchiveBytes
              ) {
                stream.destroy();
                fail("ZIP_BOMB");
              }
            });
            stream.on("error", () => fail("ARCHIVE_INVALID"));
            stream.on("end", () => {
              if (!settled) zip.readEntry();
            });
          });
        });
        zip.readEntry();
      },
    );
  });
}

async function extract(request: WorkerRequest): Promise<ExtractionResult> {
  const media = await inspectArchive(request.path, request);
  // Loading a native binding is itself isolated and only happens after refusal guards.
  const require = createRequire(import.meta.url);
  let anydoc: typeof import("@firecrawl/anydoc");
  try {
    anydoc = require("@firecrawl/anydoc");
  } catch {
    throw new ExtractionError("PARSER_UNAVAILABLE");
  }
  const bytes = await readFile(request.path);
  let text: string;
  try {
    text = await anydoc.toMarkdownBytes(
      bytes,
      "docx" as import("@firecrawl/anydoc").Format,
    );
  } catch {
    throw new ExtractionError("PARSE_FAILED");
  }
  if (!text.trim()) throw new ExtractionError("NO_DOCUMENT_TEXT");
  if (Buffer.byteLength(text) > request.maxOutputBytes)
    throw new ExtractionError("PARSER_OUTPUT_LIMIT");
  return {
    profile: "document-v1",
    text,
    format: "markdown",
    completeness: media ? "partial" : "complete",
    may_omit_content: media,
    pages_needing_ocr: [],
    truncated: false,
    parser: { name: "anydoc", version: "0.1.3" },
  };
}

if (import.meta.main) {
  let maxOutput = 15 * 1024 * 1024;
  let serialized: string;
  try {
    const request = requestSchema.parse(JSON.parse(await Bun.stdin.text()));
    maxOutput = request.maxOutputBytes;
    serialized = JSON.stringify({ ok: true, result: await extract(request) });
    if (Buffer.byteLength(serialized) > maxOutput)
      throw new ExtractionError("PARSER_OUTPUT_LIMIT");
  } catch (error) {
    serialized = JSON.stringify({
      ok: false,
      code: error instanceof ExtractionError ? error.code : "PARSE_FAILED",
    });
  }
  await Bun.write(Bun.stdout, serialized);
}
