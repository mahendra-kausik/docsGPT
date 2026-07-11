import type { SseFrame } from "./types.ts";

// EventSource is GET-only, but /ask/stream is a POST — so we read the body stream ourselves and
// parse SSE frames (blocks separated by a blank line; `event:` and `data:` lines within).
export async function* streamAsk(
  apiUrl: string,
  body: { question: string; synthesis_model?: string },
): AsyncGenerator<SseFrame> {
  const resp = await fetch(`${apiUrl}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let sep: number;
    // A complete frame ends at a blank line; keep the trailing partial in buf.
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const block = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      let event = "message";
      const dataLines: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length) {
        yield { event, data: JSON.parse(dataLines.join("\n")) } as SseFrame;
      }
    }
  }
}
