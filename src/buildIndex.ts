import dotenv from "dotenv";
import { readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import OpenAI from "openai";
import pdfParse from "pdf-parse";
import mammoth from "mammoth";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

dotenv.config();

const DATA_DIR = path.join(process.cwd(), "data");
const INDEX_DIR = path.join(process.cwd(), "index");

const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const CHUNK_SIZE_CHARS = parseInt(process.env.CHUNK_SIZE_CHARS || "1000", 10);
const CHUNK_OVERLAP_CHARS = parseInt(
  process.env.CHUNK_OVERLAP_CHARS || "120",
  10
);

type Domain = "resume" | "personal";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function readTextFromPath(p: string): Promise<string> {
  const ext = path.extname(p).toLowerCase();
  if (ext === ".pdf") {
    const buf = await readFile(p);
    const parsed = await pdfParse(buf);
    return parsed.text || "";
  } else if (ext === ".docx") {
    const buf = await readFile(p);
    const res = await mammoth.extractRawText({ buffer: buf });
    return res.value || "";
  } else if (ext === ".md" || ext === ".txt") {
    return await readFile(p, "utf8");
  }
  throw new Error(`Unsupported file type: ${ext}`);
}

async function loadDocuments(): Promise<
  Array<{ text: string; source: string; domain: Domain }>
> {
  const bases: Array<{ base: string; domain: Domain }> = [
    { base: "resume", domain: "resume" },
    { base: "profile", domain: "personal" },
    { base: "personal", domain: "personal" },
    { base: "about", domain: "personal" },
    { base: "bio", domain: "personal" },
    { base: "interests", domain: "personal" },
  ];
  const exts = [".pdf", ".docx", ".md", ".txt"];

  const docs: Array<{ text: string; source: string; domain: Domain }> = [];
  for (const { base, domain } of bases) {
    for (const ext of exts) {
      const p = path.join(DATA_DIR, `${base}${ext}`);
      if (!existsSync(p)) continue;
      const text = await readTextFromPath(p);
      docs.push({ text, source: path.basename(p), domain });
      break; // prefer the first matching extension for a given base
    }
  }
  if (docs.length === 0) {
    throw new Error(
      "Place files under data/: resume.(pdf|docx|md|txt) and optionally profile|personal|about|bio|interests with supported extensions."
    );
  }
  return docs;
}

async function chunkText(text: string): Promise<string[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE_CHARS,
    chunkOverlap: CHUNK_OVERLAP_CHARS,
    separators: ["\n\n", "\n", ". ", " ", ""],
  });
  const chunks = await splitter.splitText(text);
  return chunks.map((c) => c.trim()).filter((c) => c.length > 50);
}

function normalize(vec: number[]): number[] {
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1e-12;
  return vec.map((v) => v / norm);
}

async function embedBatch(texts: string[]): Promise<number[][]> {
  const resp = await client.embeddings.create({
    model: EMBED_MODEL,
    input: texts,
  });
  return resp.data.map((d) => normalize(d.embedding as number[]));
}

export async function buildIndex() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY not set. Create .env from .env.example");
  }

  const docs = await loadDocuments();
  for (const d of docs) {
    if (!d.text || d.text.trim().length < 100) {
      console.warn(
        `Warning: extracted text from ${d.source} seems short. PDF extraction may be lossy. Consider DOCX/MD.`
      );
    }
  }
  const totalChars = docs.reduce((s, d) => s + d.text.length, 0);
  console.log(`Extracted characters (all docs): ${totalChars}`);

  // Chunk each document and keep metadata
  const chunkTexts: string[] = [];
  const metas: Array<{ source: string; domain: Domain }> = [];
  for (const d of docs) {
    const chunks = await chunkText(d.text);
    console.log(`${d.source} (${d.domain}) -> ${chunks.length} chunks`);
    for (const ch of chunks) {
      chunkTexts.push(ch);
      metas.push({ source: d.source, domain: d.domain });
    }
  }
  console.log(`Total chunks: ${chunkTexts.length}`);
  console.log(
    "First chunk lengths:",
    chunkTexts.slice(0, 5).map((c) => c.length)
  );

  // Embed in batches
  const BATCH = 64;
  const embeddings: number[][] = [];
  for (let i = 0; i < chunkTexts.length; i += BATCH) {
    const batch = chunkTexts.slice(i, i + BATCH);
    const vecs = await embedBatch(batch);
    embeddings.push(...vecs);
    process.stdout.write(
      `Embedded ${Math.min(i + BATCH, chunkTexts.length)}/${
        chunkTexts.length
      }\r`
    );
  }
  console.log(
    `\nEmbeddings shape: [${embeddings.length}, ${embeddings[0]?.length || 0}]`
  );

  const records = chunkTexts.map((ch, i) => ({
    id: i,
    text: ch,
    source: metas[i].source,
    domain: metas[i].domain,
    embedding: embeddings[i],
  }));

  await writeFile(
    path.join(INDEX_DIR, "knowledge.index.json"),
    JSON.stringify(records, null, 2),
    "utf8"
  );
  console.log("Saved index at index/knowledge.index.json");
}

// Allow running this file directly via `npm run index:build`
if (process.argv[1] && process.argv[1].endsWith("buildIndex.ts")) {
  buildIndex().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
