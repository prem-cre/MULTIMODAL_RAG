"""
run_local.py  –  Clean-start the FastAPI backend for local development.

Usage (from the multimodal_RAG folder, with venv active):
    python run_local.py

Then in a SEPARATE terminal:
    npm run dev          →  http://localhost:3000
"""
import os
import shutil
import tempfile

# ── 1. Delete old chroma_db on disk (holds stale settings from previous runs) ──
chroma_path = os.path.join(tempfile.gettempdir(), "chroma_db")
if os.path.exists(chroma_path):
    shutil.rmtree(chroma_path, ignore_errors=True)
    print(f"✅  Deleted stale chroma dir: {chroma_path}")
else:
    print(f"ℹ️   No stale chroma dir found at {chroma_path}")

# ── 2. Delete pycache so Python re-compiles from the latest .py source ─────────
for root, dirs, files in os.walk("."):
    # Skip node_modules / .git
    dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", ".next", "venv")]
    for d in dirs:
        if d == "__pycache__":
            full = os.path.join(root, d)
            shutil.rmtree(full, ignore_errors=True)
            print(f"✅  Deleted pycache: {full}")

# ── 3. Wipe chromadb in-process cache just in case ────────────────────────────
try:
    from chromadb.api.client import SharedSystemClient
    SharedSystemClient.clear_system_cache()
    print("✅  Cleared SharedSystemClient cache")
except Exception:
    pass

# ── 4. Start uvicorn ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  Multimodal RAG  –  FastAPI backend")
print("  http://127.0.0.1:8000")
print()
print("  Start Next.js in a SEPARATE terminal:")
print("    npm run dev   →  http://localhost:3000")
print("=" * 60)
print()

import uvicorn
uvicorn.run(
    "api.index:app",
    host="127.0.0.1",
    port=8000,
    reload=False,   # reload=True can cause double-import; keep False for stability
)
