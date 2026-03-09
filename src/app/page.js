"use client";

import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("chat");
  const [lastSources, setLastSources] = useState([]);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError("");
  };

  /**
   * Safe response reader — fixes "body stream already read" error.
   *
   * Root cause: calling response.json() consumes the body stream. If json()
   * throws (e.g. the server returned plain text / HTML), a subsequent call to
   * response.text() fails because the stream is already closed.
   *
   * Fix: always read as text first, then attempt JSON.parse().
   */
  const readResponseSafely = async (response) => {
    const text = await response.text();           // read ONCE
    try {
      return { json: JSON.parse(text), text };    // try to parse
    } catch {
      return { json: null, text };                // plain text (e.g. HTML error page)
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsProcessing(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    console.log("🚀 Starting upload to /api/ingest...");
    try {
      const response = await fetch("/api/ingest", {
        method: "POST",
        body: formData,
      });

      console.log("📡 API Response received. Status:", response.status);
      const { json, text } = await readResponseSafely(response);
      console.log("📦 Parsed response:", json || text);

      if (!response.ok) {
        const msg =
          json?.detail ||
          json?.message ||
          (response.status === 504 ? "Vercel Execution Timeout (10s reached). Try a smaller PDF." :
            response.status === 413 ? "File too large for Vercel Hobby (Max 4.5MB)." :
              (text.length < 300 ? text : `Server Error: ${response.status}`));

        console.error("❌ Upload Error Details:", msg);
        throw new Error(msg);
      }

      setMessages([
        {
          role: "ai",
          content: `Excellent! **${file.name}** has been indexed. What would you like to know about it?`,
        },
      ]);
      setActiveTab("chat");
    } catch (err) {
      console.error("🔥 Global Catch - Upload Failed:", err);
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    const currentQuery = query;
    console.log("✍️ User Query:", currentQuery);
    setMessages((prev) => [...prev, { role: "user", content: currentQuery }]);
    setQuery("");
    setIsLoading(true);

    const formData = new FormData();
    formData.append("query", currentQuery);

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        body: formData,
      });

      console.log("📡 Query Response Status:", response.status);
      const { json, text } = await readResponseSafely(response);

      if (!response.ok) {
        const msg =
          json?.detail ||
          json?.message ||
          (text.length < 300 ? text : `Query failed: ${response.status}`);

        console.error("❌ Query Error:", msg);
        throw new Error(msg);
      }

      setMessages((prev) => [
        ...prev,
        { role: "ai", content: json?.answer || "No answer returned." },
      ]);
      setLastSources(json?.chunks || []);
    } catch (err) {
      console.error("🔥 Global Catch - Query Failed:", err);
      setMessages((prev) => [
        ...prev,
        { role: "ai", content: `❌ **Error:** ${err.message}` },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="glass-card">
        <header
          style={{
            marginBottom: "2rem",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <h1>Multimodal RAG</h1>
            <p>Powered by Gemini &amp; Unstructured</p>
          </div>

          <div
            style={{
              display: "flex",
              background: "rgba(255,255,255,0.05)",
              borderRadius: "12px",
              padding: "0.3rem",
            }}
          >
            {["ingest", "chat", "sources"].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  background:
                    activeTab === tab ? "rgba(69,162,158,0.2)" : "none",
                  border: "none",
                  color: activeTab === tab ? "#45A29E" : "#8892b0",
                  padding: "0.5rem 1rem",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontWeight: "bold",
                  textTransform: "capitalize",
                }}
              >
                {tab === "ingest" ? "Upload" : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
        </header>

        {/* ─── Upload tab ─────────────────────────────────────────────────── */}
        {activeTab === "ingest" && (
          <div style={{ animation: "slideUp 0.4s ease-out" }}>
            <div className="upload-section">
              <input
                type="file"
                id="file-upload"
                hidden
                onChange={handleFileChange}
                accept=".pdf"
              />
              <label htmlFor="file-upload" style={{ cursor: "pointer" }}>
                <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>📄</div>
                <h3>{file ? file.name : "Choose a PDF Document"}</h3>
                <p>
                  {file ? "Ready to index" : "Drag and drop or click to browse"}
                </p>
              </label>
            </div>

            {error && (
              <div
                style={{
                  color: "#ff4b2b",
                  marginBottom: "1rem",
                  textAlign: "center",
                  wordBreak: "break-word",
                }}
              >
                ⚠️ {error}
              </div>
            )}

            <button
              className="btn-primary"
              onClick={handleUpload}
              disabled={!file || isProcessing}
              style={{ width: "100%", justifyContent: "center" }}
            >
              {isProcessing ? "🔍 Indexing…" : "⚡ Index Multimodal Content"}
            </button>
          </div>
        )}

        {/* ─── Chat tab ────────────────────────────────────────────────────── */}
        {activeTab === "chat" && (
          <div style={{ display: "flex", flexDirection: "column", height: "500px" }}>
            <div
              style={{
                flex: 1,
                overflowY: "auto",
                marginBottom: "1.5rem",
                paddingRight: "0.5rem",
              }}
            >
              {messages.length === 0 ? (
                <div
                  style={{
                    textAlign: "center",
                    marginTop: "4rem",
                    color: "#8892b0",
                  }}
                >
                  <div style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>
                    🤖
                  </div>
                  <h3>Your Intelligent Research Assistant</h3>
                  <p>
                    Upload a document to start a conversation about text, tables,
                    and images.
                  </p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`chat-bubble ${msg.role === "user" ? "user-bubble" : "ai-bubble"
                      }`}
                  >
                    <div style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>

            <form
              onSubmit={handleQuery}
              style={{ position: "relative" }}
            >
              <input
                type="text"
                placeholder="Ask a question about the document…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={isLoading}
              />
              <button
                type="submit"
                className="btn-primary"
                style={{
                  position: "absolute",
                  right: "8px",
                  top: "8px",
                  padding: "0.6rem 1rem",
                }}
                disabled={isLoading || !query.trim()}
              >
                {isLoading ? "…" : "→"}
              </button>
            </form>
          </div>
        )}

        {/* ─── Sources tab ─────────────────────────────────────────────────── */}
        {activeTab === "sources" && (
          <div
            style={{
              animation: "slideUp 0.4s ease-out",
              maxHeight: "500px",
              overflowY: "auto",
            }}
          >
            {lastSources.length === 0 ? (
              <div
                style={{
                  textAlign: "center",
                  marginTop: "4rem",
                  color: "#8892b0",
                }}
              >
                <p>
                  Retrieved segments will appear here after your first question.
                </p>
              </div>
            ) : (
              lastSources.map((source, idx) => (
                <div
                  key={idx}
                  className="chat-bubble"
                  style={{ marginBottom: "1.5rem" }}
                >
                  <div
                    style={{
                      fontWeight: "bold",
                      color: "#45A29E",
                      marginBottom: "0.5rem",
                    }}
                  >
                    Source Segment {idx + 1}
                  </div>
                  <div style={{ fontSize: "0.9rem", lineHeight: "1.5" }}>
                    {source.page_content}
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      <footer
        style={{ marginTop: "2rem", color: "#8892b0", fontSize: "0.8rem" }}
      >
        Professional Deployment Hub ⚡ Vercel + Gemini
      </footer>
    </div>
  );
}
