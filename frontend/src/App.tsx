import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { ApiError, askChat, type Source } from "./api";
import "./App.css";

interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  error?: boolean;
}

let nextId = 0;

/** A lone `$` before a digit is currency, not math.
 *
 * remark-math treats single dollars as inline delimiters, so "billed at $47
 * per index and $12 per million" would otherwise be parsed as one math span
 * and rendered as mangled glyphs. Escaping the opening dollar keeps it literal.
 * `$$` is left alone so display math starting with a digit still works, and
 * real inline math ($P_i$, $n$, $\{p_0\}$) is untouched because it does not
 * start with a digit.
 */
function protectCurrency(markdown: string): string {
  return markdown.replace(/(\$\$?)(?=\d)/g, (match) =>
    match === "$$" ? match : "\\$",
  );
}

type Theme = "light" | "dark";
const THEME_KEY = "askmynotes-theme";

/** Saved choice if there is one, otherwise whatever the OS prefers. */
function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // localStorage can throw in private mode or with site data blocked.
  }
  // Guarded for non-browser contexts (SSR, tests); in the app window is always
  // present, so this falls through to the real system preference.
  if (typeof window === "undefined" || !window.matchMedia) return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function SourceList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState<number | null>(null);

  return (
    <ul className="sources">
      {sources.map((source, i) => (
        <li key={`${source.source}-${source.page}-${i}`}>
          <button
            type="button"
            className="source-toggle"
            onClick={() => setOpen(open === i ? null : i)}
            aria-expanded={open === i}
          >
            <span className="source-index">[{i + 1}]</span>
            <span className="source-name">
              {source.source} · p.{source.page}
            </span>
            <span className="source-chevron">{open === i ? "−" : "+"}</span>
          </button>
          {open === i && <p className="source-excerpt">{source.excerpt}</p>}
        </li>
      ))}
    </ul>
  );
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // Choice simply will not persist; the UI still switches.
    }
  }, [theme]);

  async function send(event: React.FormEvent) {
    event.preventDefault();
    const query = input.trim();
    if (!query || pending) return;

    setMessages((prev) => [...prev, { id: nextId++, role: "user", text: query }]);
    setInput("");
    setPending(true);

    try {
      const data = await askChat(query);
      setMessages((prev) => [
        ...prev,
        {
          id: nextId++,
          role: "assistant",
          text: data.answer,
          sources: data.sources,
        },
      ]);
    } catch (err) {
      const retry =
        err instanceof ApiError && err.retryAfter
          ? ` Try again in ${err.retryAfter}s.`
          : "";
      setMessages((prev) => [
        ...prev,
        {
          id: nextId++,
          role: "assistant",
          text: (err as Error).message + retry,
          error: true,
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Ask My Notes</h1>
          <p>Answers grounded in your indexed notes.</p>
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? "☀" : "☾"}
        </button>
      </header>

      <main className="history">
        {messages.length === 0 && !pending && (
          <p className="empty">
            Ask something about your notes — the model will search them, and do
            any arithmetic, on its own.
          </p>
        )}

        {messages.map((message) => (
          <article
            key={message.id}
            className={`message ${message.role}${message.error ? " error" : ""}`}
          >
            <div className="bubble">
              {message.role === "assistant" && !message.error ? (
                // Answers come back as markdown - headings, bold, lists - so
                // they are rendered rather than shown with literal asterisks.
                // react-markdown escapes HTML by default, so model output
                // cannot inject markup.
                <div className="markdown">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                  >
                    {protectCurrency(message.text)}
                  </ReactMarkdown>
                </div>
              ) : (
                message.text
              )}
            </div>
            {message.sources && message.sources.length > 0 && (
              <SourceList sources={message.sources} />
            )}
          </article>
        ))}

        {pending && (
          <article className="message assistant">
            <div className="bubble thinking">
              <span />
              <span />
              <span />
            </div>
          </article>
        )}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={send}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask your notes…"
          aria-label="Your question"
          disabled={pending}
          autoFocus
        />
        <button type="submit" disabled={pending || !input.trim()}>
          {pending ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}
