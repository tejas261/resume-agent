import "dotenv/config";
import express from "express";
import cors from "cors";
import morgan from "morgan";
import { answer, retrieveChunks, indexInfo } from "./qaLib";
import { buildIndex } from "./buildIndex";

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;
const TOP_K = Number(process.env.TOP_K || "5");

app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use(morgan("dev"));

app.get("/health", (_req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

// app.get("/index-info", async (_req, res) => {
//   try {
//     const info = await indexInfo();
//     res.json(info);
//   } catch (err: any) {
//     res.status(500).json({ error: err?.message || "index-info failed" });
//   }
// });

// app.post("/retrieve", async (req, res) => {
//   try {
//     const { query, domain } = req.body || {};
//     if (!query || typeof query !== "string") {
//       return res.status(400).json({ error: "query (string) is required" });
//     }
//     if (domain && !["resume", "personal"].includes(domain)) {
//       return res
//         .status(400)
//         .json({ error: "domain must be 'resume' or 'personal'" });
//     }
//     const results = await retrieveChunks(query, TOP_K, domain);
//     res.json({ query, k: TOP_K, domain: domain || "all", results });
//   } catch (err: any) {
//     res.status(500).json({ error: err?.message || "retrieve failed" });
//   }
// });

app.post("/chat", async (req, res) => {
  try {
    const { question } = req.body || {};
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "question (string) is required" });
    }
    const result = await answer(question, TOP_K);
    res.json({
      question,
      answer: result.output,
    });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || "Chat API failed" });
  }
});

app.use((_req, res) => res.status(404).json({ error: "not found" }));

// eslint-disable-next-line @typescript-eslint/no-unused-vars
app.use(
  (
    err: any,
    _req: express.Request,
    res: express.Response,
    _next: express.NextFunction
  ) => {
    console.error(err);
    res.status(500).json({ error: "internal error" });
  }
);

async function start() {
  try {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY not set. Create .env from .env.example");
    }
    console.log("[startup] Building index...");
    await buildIndex();
    console.log("[startup] Index built successfully.");

    app.listen(PORT, () => {
      console.log(`Server listening on http://localhost:${PORT}`);
    });
  } catch (err: any) {
    console.error("Failed to start server:", err?.message || err);
    process.exit(1);
  }
}

start();

// cURL
// curl -X POST http://localhost:3000/chat -H "Content-Type: application/json" -d '{"question":"What do you work on at Fynd?"}'
