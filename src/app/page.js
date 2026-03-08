"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";

export default function Home() {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("chat");
  const [lastSources, setLastSources] = useState([]);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsProcessing(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/ingest", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Upload failed. Verify your API keys.");
      }

      await response.json();
      setMessages([{ role: "ai", content: `Excellent! **${file.name}** has been indexed. What would you like to know about it?` }]);
      setActiveTab("chat");
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    const userMsg = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setQuery("");
    setIsLoading(true);

    const formData = new FormData();
    formData.append("query", query);

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Query failed. Ensure you ingested a document.");
      }

      const data = await response.json();
      setMessages((prev) => [...prev, { role: "ai", content: data.answer }]);
      setLastSources(data.chunks || []);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "ai", content: `❌ **Error:** ${err.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="glass-card">
        <header style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>Multimodal RAG</h1>
            <p>Powered by Gemini & Unstructured</p>
          </div>

          <div style={{ display: 'flex', background: 'rgba(255,255,255,0.05)', borderRadius: '12px', padding: '0.3rem' }}>
            <button
              onClick={() => setActiveTab('ingest')}
              style={{
                background: activeTab === 'ingest' ? 'rgba(69, 162, 158, 0.2)' : 'none',
                border: 'none', color: activeTab === 'ingest' ? '#45A29E' : '#8892b0',
                padding: '0.5rem 1rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold'
              }}
            >
              Upload
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              style={{
                background: activeTab === 'chat' ? 'rgba(69, 162, 158, 0.2)' : 'none',
                border: 'none', color: activeTab === 'chat' ? '#45A29E' : '#8892b0',
                padding: '0.5rem 1rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold'
              }}
            >
              Chat
            </button>
            <button
              onClick={() => setActiveTab('sources')}
              style={{
                background: activeTab === 'sources' ? 'rgba(69, 162, 158, 0.2)' : 'none',
                border: 'none', color: activeTab === 'sources' ? '#45A29E' : '#8892b0',
                padding: '0.5rem 1rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold'
              }}
            >
              Sources
            </button>
          </div>
        </header>

        {activeTab === 'ingest' && (
          <div style={{ animation: 'slideUp 0.4s ease-out' }}>
            <div className="upload-section">
              <input
                type="file"
                id="file-upload"
                hidden
                onChange={handleFileChange}
                accept=".pdf"
              />
              <label htmlFor="file-upload" style={{ cursor: 'pointer' }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📄</div>
                <h3>{file ? file.name : "Choose a PDF Document"}</h3>
                <p>{file ? "Ready to index" : "Drag and drop or click to browse"}</p>
              </label>
            </div>

            {error && <div style={{ color: '#ff4b2b', marginBottom: '1rem', textAlign: 'center' }}>⚠️ {error}</div>}

            <button
              className="btn-primary"
              onClick={handleUpload}
              disabled={!file || isProcessing}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              {isProcessing ? "🔍 Indexing..." : "⚡ Index Multimodal Content"}
            </button>
          </div>
        )}

        {activeTab === 'chat' && (
          <div style={{ display: 'flex', flexDirection: 'column', height: '500px' }}>
            <div style={{ flex: 1, overflowY: 'auto', marginBottom: '1.5rem', paddingRight: '0.5rem' }}>
              {messages.length === 0 ? (
                <div style={{ textAlign: 'center', marginTop: '4rem', color: '#8892b0' }}>
                  <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>🤖</div>
                  <h3>Your Intelligent Research Assistant</h3>
                  <p>Upload a document to start a conversation about text, tables, and images.</p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} className={`chat-bubble ${msg.role === 'user' ? 'user-bubble' : 'ai-bubble'}`}>
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleQuery} style={{ position: 'relative' }}>
              <input
                type="text"
                placeholder="Ask a question about the papers..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={isLoading}
              />
              <button
                type="submit"
                className="btn-primary"
                style={{ position: 'absolute', right: '8px', top: '8px', padding: '0.6rem 1rem' }}
                disabled={isLoading || !query.trim()}
              >
                {isLoading ? "..." : "→"}
              </button>
            </form>
          </div>
        )}

        {activeTab === 'sources' && (
          <div style={{ animation: 'slideUp 0.4s ease-out', maxHeight: '500px', overflowY: 'auto' }}>
            {lastSources.length === 0 ? (
              <div style={{ textAlign: 'center', marginTop: '4rem', color: '#8892b0' }}>
                <p>Retrieved segments will appear here after your first question.</p>
              </div>
            ) : (
              lastSources.map((source, idx) => (
                <div key={idx} className="chat-bubble" style={{ marginBottom: '1.5rem' }}>
                  <div style={{ fontWeight: 'bold', color: '#45A29E', marginBottom: '0.5rem' }}>Source Segment {idx + 1}</div>
                  <div style={{ fontSize: '0.9rem', lineHeight: '1.5' }}>{source.page_content}</div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      <footer style={{ marginTop: '2rem', color: '#8892b0', fontSize: '0.8rem' }}>
        Professional Deployment Hub ⚡ Vercel + Gemini
      </footer>
    </div>
  );
}
