import "dotenv/config";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import OpenAI from "openai/index.mjs";

const KNOWLEDGE_PATH = path.join(
  process.cwd(),
  "index",
  "knowledge.index.json"
);
const FALLBACK_RESUME_PATH = path.join(
  process.cwd(),
  "index",
  "resume.index.json"
);
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const CHAT_MODEL = process.env.CHAT_MODEL || "gpt-4o-mini";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

type Domain = "resume" | "personal";

export type Citation = { cid: string; text: string };
export type QAResult = { output: string; used: Citation[] };

export type RecordT = {
  id: number;
  text: string;
  source: string;
  domain?: Domain;
  embedding: number[];
};

export type RetrieveOptions = { domainFilter?: Domain; ensureTerms?: string[] };

function cosine(a: number[], b: number[]): number {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

function normalize(vec: number[]): number[] {
  const n = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1e-12;
  return vec.map((v) => v / n);
}

function resolveIndexPath(): string {
  if (existsSync(KNOWLEDGE_PATH)) return KNOWLEDGE_PATH;
  return FALLBACK_RESUME_PATH;
}

async function loadIndex(): Promise<RecordT[]> {
  const pathToUse = resolveIndexPath();
  const raw = await readFile(pathToUse, "utf8");
  const payload = JSON.parse(raw) as RecordT[];
  if (!Array.isArray(payload) || payload.length === 0) {
    throw new Error("Index is empty. Run: npm run index:build");
  }
  return payload;
}

function ensureTermsForQuestion(q: string): string[] {
  const s = q.toLowerCase();
  const ensures: string[] = [];
  if (s.includes("fynd")) {
    ensures.push("ratl.ai", "about ratl.ai", "ratl ");
  }
  return ensures;
}

function classifyQuestion(q: string): Domain | "both" | "unknown" {
  const s = q.toLowerCase();
  const personalKeys = [
    "hobby",
    "hobbies",
    "interest",
    "interests",
    "free time",
    "outside work",
    "outside of work",
    "weekend",
    "leisure",
    "music",
    "books",
    "reading",
    "travel",
    "sports",
    "volunteer",
    "volunteering",
    "hike",
    "gaming",
    "family",
    "pets",
    "art",
    "photography",
  ];
  const resumeKeys = [
    "resume",
    "cv",
    "experience",
    "work",
    "role",
    "title",
    "employer",
    "company",
    "education",
    "skill",
    "project",
    "years",
    "responsibil",
    "achievement",
  ];
  const isPersonal = personalKeys.some((k) => s.includes(k));
  const isResume = resumeKeys.some((k) => s.includes(k));
  if (isPersonal && isResume) return "both";
  if (isPersonal) return "personal";
  if (isResume) return "resume";
  return "unknown";
}

async function retrieve(
  query: string,
  k = 5,
  opts: RetrieveOptions = {}
): Promise<Array<RecordT & { score: number; cid: string }>> {
  const payload = await loadIndex();
  const filtered = opts.domainFilter
    ? payload.filter((r) => (r.domain || "resume") === opts.domainFilter)
    : payload;
  if (filtered.length === 0) {
    return [];
  }
  const q = (
    await client.embeddings.create({ model: EMBED_MODEL, input: query })
  ).data[0].embedding as number[];
  const qnorm = normalize(q);

  const scored = filtered.map((rec) => ({
    ...rec,
    score: cosine(qnorm, rec.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);

  const ensure = (opts.ensureTerms || []).map((t) => t.toLowerCase());
  let ensured: Array<RecordT & { score: number }> = [];
  if (ensure.length > 0) {
    const matches = scored.filter((r) =>
      ensure.some((t) => r.text.toLowerCase().includes(t))
    );
    const seen = new Set<number>();
    ensured = matches.filter((m) => {
      if (seen.has(m.id)) return false;
      seen.add(m.id);
      return true;
    });
  }

  const seen = new Set<number>(ensured.map((e) => e.id));
  const rest = scored.filter((r) => !seen.has(r.id));
  const final = [...ensured, ...rest].slice(0, k);
  return final.map((r, i) => ({ ...r, cid: `C${i + 1}` }));
}

function buildSystemPrompt(domainHint: Domain | "both" | "unknown") {
  const scope =
    domainHint === "personal"
      ? "personal profile"
      : domainHint === "resume"
      ? "professional resume"
      : "provided context";
  return (
    `You are a concise, highly professional assistant for answering questions about the user based on their ${scope}.\n` +
    "- Use ONLY the provided context for any facts about the user. If information is missing or ambiguous, say so and ask a clarifying question.\n" +
    "- Write in a formal, clear tone. Return at most two bullet points, each one sentence.\n" +
    "- Cite supporting snippets using bracketed citations like [C1], [C2], matching the provided context IDs.\n" +
    "- Do NOT invent dates, titles, employers, or personal details. If unsure, request clarification.\n"
  );
}

function formatContext(ctx: Array<{ cid: string; text: string }>): string {
  return ctx.map((c) => `[${c.cid}] ${c.text}`).join("\n\n");
}

export async function answer(question: string, k = 5): Promise<QAResult> {
  const domainHint = classifyQuestion(question);
  const ensure = ensureTermsForQuestion(question);
  let top: Array<RecordT & { score: number; cid: string }> = [];
  if (ensure.length > 0) {
    top = await retrieve(question, k, { ensureTerms: ensure });
  } else if (domainHint === "personal") {
    top = await retrieve(question, k, { domainFilter: "personal" });
    if (top.length === 0) {
      top = await retrieve(question, k);
    }
  } else if (domainHint === "resume") {
    top = await retrieve(question, k, { domainFilter: "resume" });
  } else {
    top = await retrieve(question, k);
  }

  const used = top.map((t) => ({ cid: t.cid, text: t.text }));
  const contextBlock = formatContext(used);

  const styleHint =
    "- Limit the answer to at most two bullet points, each one sentence.\n";
  const userPrompt = `Context (snippets):\n\n${contextBlock}\n\nUser question: ${question}\n\nInstructions:\n- Answer using only the context for user-specific facts.\n- Include relevant citations [C#] after the sentences they support.\n- If the context does not contain the requested fact, say so and ask for the missing detail.\n${styleHint}`;

  const resp = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      { role: "system", content: buildSystemPrompt(domainHint) },
      { role: "user", content: userPrompt },
    ],
    temperature: 0.2,
  });
  const output = resp.choices[0]?.message?.content || "";
  return { output, used };
}
