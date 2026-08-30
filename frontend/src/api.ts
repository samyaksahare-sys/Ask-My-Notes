// Types mirror backend/main.py: AskResponse / Source.

export interface Source {
  source: string;
  page: number;
  distance: number;
  excerpt: string;
}

export interface ChatResponse {
  answer: string;
  refused: boolean;
  sources: Source[];
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

/** Error carrying what the backend told us, so the UI can show it verbatim. */
export class ApiError extends Error {
  status: number;
  retryAfter: number | null;

  constructor(message: string, status: number, retryAfter: number | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

export async function askChat(
  query: string,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    // fetch only rejects on network/CORS failure, never on a 4xx/5xx.
    throw new ApiError(
      `Cannot reach the API at ${API_BASE}. Is the backend running?`,
      0,
      null,
    );
  }

  if (!response.ok) {
    const retryAfter = Number(response.headers.get("Retry-After")) || null;
    let detail = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      // FastAPI puts HTTPException messages in `detail`; validation errors
      // put an array of issues there instead.
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) {
        detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
      }
    } catch {
      // Non-JSON error body; keep the status-based message.
    }
    throw new ApiError(detail, response.status, retryAfter);
  }

  return (await response.json()) as ChatResponse;
}
