// Mirrors _answer_payload / _citations_payload in src/api/app.py.

export interface Citation {
  marker: string;
  chunk_id: string;
  heading_path: string;
  source_url: string;
}

export interface AskDone {
  answer: string;
  grounded: boolean;
  retries: number;
  citations: Citation[];
  invalid_citations: string[];
  metrics: Record<string, number>;
}

// One decoded SSE frame: {event, data}. Matches _stage_event / token / done / error.
export type SseFrame =
  | { event: "stage"; data: { stage: string; chunks?: number; grounded?: boolean; n?: number } }
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: AskDone }
  | { event: "error"; data: { detail: string } };
