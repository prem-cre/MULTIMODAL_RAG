# 🚀 Multimodal RAG Deployment (Vercel Edition)

Your application has been architecturaly transformed for **Vercel's Serverless environment**. This fixes the common issues with Python dependencies (Tesseract, Poppler, `sentence-transformers`) being too heavy for serverless runtimes.

## 🏗️ New Architecture
- **Frontend**: Next.js (Premium React UI, stunning glassmorphic design).
- **Backend**: FastAPI (Python Serverless Function in `/api`).
- **Partitioning**: Remote OCR via **Unstructured Cloud API** (No binaries needed!).
- **Embeddings**: Cloud-based **Google Gemini Embeddings** (Low memory, high speed).

---

## 🔑 Required API Keys
To run the cloud version, you MUST add these to your **Vercel Environment Variables**:
1. `GEMINI_API_KEY`: Your Google Gemini key.
2. `UNSTRUCTURED_API_KEY`: Get a free key at [Unstructured.io](https://unstructured.io/api-key-free).
3. `UNSTRUCTURED_API_URL`: Use `https://api.unstructured.io/general/v0/general`.

---

## 🚀 How to Deploy in 2 Minutes
1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Transform to Next.js + FastAPI for Vercel"
   git push origin main
   ```
2. **Connect to Vercel**: 
   - Go to [Vercel.com](https://vercel.com).
   - Click **Add New** → **Project**.
   - Import this GitHub repository.
   - Vercel will automatically detect the Next.js frontend and the `/api` backend.
3. **Add Environment Variables**:
   - In the Vercel dashboard, go to **Settings** → **Environment Variables**.
   - Add the 3 keys mentioned above.
4. **Deploy**: Click **Deploy**. Done.

---

## 🛠️ Performance Notes
- **Hi-Res Partitioning**: Since OCR now happens on Unstructured's servers, your local machine (or the Vercel serverless function) won't crash.
- **Cold Starts**: The first query might take ~5-10 seconds because the Python function needs to "warm up". Subsequent queries will be nearly instant.
- **Temporary Storage**: The Vector DB is stored in Vercel's `/tmp` folder. It will persist for the duration of the lambda's life (usually ~15-30 mins of inactivity before it's wiped). For long-term persistence, you should connect a hosted database like Pinecone.

Professional RAG system, Ready for Production. ⚡
