# 🚀 Multimodal RAG Deployment (Vercel Edition)

Your application has been architecturaly transformed for **Vercel's Serverless environment**. This fixes the common issues with Python dependencies (Tesseract, Poppler, `sentence-transformers`) by using **Cloud Services**.

## 🏗️ New Architecture
- **Frontend**: Next.js (Premium React UI).
- **Backend**: FastAPI (Serverless in `/api`).
- **Partitioning & Chunking**: Remote via **Unstructured Cloud API** (Zero local binaries!).
- **Embeddings**: Cloud-based **Google Gemini Embeddings** (Low memory, high speed).

---

## 🔑 Required API Keys
Add these to your **Vercel Environment Variables**:
1. `GEMINI_API_KEY`: Your Google Gemini key.
2. `UNSTRUCTURED_API_KEY`: Get a free key at [Unstructured.io](https://unstructured.io/api-key-free).
3. `UNSTRUCTURED_API_URL`: Use `https://api.unstructuredapp.io/general/v0/general` (Optimized for better DNS resolution).

---

## 🚨 Troubleshooting Common Errors
### **1. "429 Resource Exhausted" (Gemini)**
If you see this, it means your **Google Gemini API Quota** has been hit.
- **Solution**: Switch to **Gemini 1.5 Flash** (already done in code!) which has higher free tier limits (15 Requests Per Minute).

### **2. "DNS Error" or "Connection Failed" (Unstructured)**
If you see "Failed to resolve host", it's usually because the server URL is incorrect or blocked by your network.
- **Solution**: We have updated the host to `api.unstructuredapp.io` which is more reliable.

### **3. "Upload Failed"**
If this happens, check your Vercel logs. I have added **tracebacks** and detailed error messages.
- Common cause: One of your API keys is missing or has an extra space at the end.

---

## 🚀 How to Deploy in 2 Minutes
1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Optimize for Vercel: Cloud-Native Partitioning & Chunking"
   git push origin main
   ```
2. **Connect to Vercel**: 
   - Vercel automatically detects Next.js.
   - Just remember to add your **Environment Variables** in the Vercel dashboard!

Professional RAG system, Ready for Production. ⚡
