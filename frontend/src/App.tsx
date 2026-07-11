import { useState } from "react";
import { streamAsk } from "./sse.ts";
import type { AskDone } from "./types.ts";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

const MODELS = [
  { value: "groq/llama-3.3-70b-versatile", label: "Groq Llama-3.3 70B" },
  { value: "gemini/gemini-2.5-flash", label: "Gemini 2.5 Flash" },
];

function stageLabel(d: { stage: string; chunks?: number; grounded?: boolean; n?: number }): string {
  switch (d.stage) {
    case "retrieve": return `retrieve (${d.chunks} chunks)`;
    case "synthesize": return "synthesize";
    case "verify": return `verify (${d.grounded ? "grounded ✓" : "not grounded ✗"})`;
    case "retry": return `retry (${d.n})`;
    case "refuse": return "refuse";
    default: return d.stage;
  }
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [model, setModel] = useState(MODELS[0].value);
  const [stages, setStages] = useState<string[]>([]);
  const [answer, setAnswer] = useState("");
  const [done, setDone] = useState<AskDone | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function ask() {
    if (!question.trim() || busy) return;
    setBusy(true);
    setStages([]);
    setAnswer("");
    setDone(null);
    setError("");
    try {
      for await (const frame of streamAsk(API_URL, { question, synthesis_model: model })) {
        if (frame.event === "stage") setStages((s) => [...s, stageLabel(frame.data)]);
        else if (frame.event === "token") setAnswer((a) => a + frame.data.text);
        else if (frame.event === "done") setDone(frame.data);
        else if (frame.event === "error") setError(frame.data.detail);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const isGemini = model.startsWith("gemini");

  return (
    <main style={{ maxWidth: 760, margin: "2rem auto", padding: "0 1rem", fontFamily: "system-ui, sans-serif", lineHeight: 1.5 }}>
      <h1 style={{ marginBottom: 0 }}>DocsGPT-Agent</h1>
      <p style={{ color: "#666", marginTop: 4 }}>Cited answers over the LangChain docs corpus.</p>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(); }}
        placeholder="Ask a question about LangChain… (Ctrl/Cmd+Enter to send)"
        rows={3}
        style={{ width: "100%", padding: 10, fontSize: 15, boxSizing: "border-box" }}
      />

      <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "8px 0" }}>
        <select value={model} onChange={(e) => setModel(e.target.value)} style={{ padding: 6 }}>
          {MODELS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
        {isGemini && <small style={{ color: "#a15c00" }}>20 req/day free cap</small>}
        <button onClick={ask} disabled={busy} style={{ marginLeft: "auto", padding: "6px 18px", fontSize: 15 }}>
          {busy ? "Asking…" : "Ask"}
        </button>
      </div>

      {stages.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "12px 0" }}>
          {stages.map((s, i) => (
            <span key={i} style={{ background: "#eef", borderRadius: 4, padding: "2px 8px", fontSize: 13 }}>{s}</span>
          ))}
        </div>
      )}

      {busy && stages.length === 0 && <p style={{ color: "#888" }}>Warming up… (cold start can take ~45s)</p>}

      {answer && <div style={{ whiteSpace: "pre-wrap", marginTop: 12 }}>{answer}</div>}

      {error && <p style={{ color: "#c00" }}>Error: {error}</p>}

      {done && done.citations.length > 0 && (
        <section style={{ marginTop: 20 }}>
          <h3>Citations</h3>
          <ul style={{ listStyle: "none", paddingLeft: 0 }}>
            {done.citations.map((c) => (
              <li key={c.marker}>
                [{c.marker}]{" "}
                <a href={c.source_url} target="_blank" rel="noreferrer">{c.heading_path || c.source_url}</a>
              </li>
            ))}
          </ul>
        </section>
      )}

      {done && done.invalid_citations.length > 0 && (
        <p style={{ color: "#a15c00" }}>Invalid citations dropped: {done.invalid_citations.join(", ")}</p>
      )}

      {done && (
        <footer style={{ color: "#888", fontSize: 13, marginTop: 16 }}>
          {done.metrics.llm_calls} LLM calls · {done.metrics.total_tokens} tokens · {done.metrics.latency_ms} ms
          {done.retries > 0 && ` · ${done.retries} retries`}
        </footer>
      )}
    </main>
  );
}
