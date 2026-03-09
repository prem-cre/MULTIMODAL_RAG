import os
import sys
import shutil
import time
from fastapi.testclient import TestClient

# Mock environmental variables for the test if not present
# (Using the keys from the User's .env file seen earlier)
os.environ["GEMINI_API_KEY"] = "AIzaSyC3_zN8jEq_K4qiVuSjPSrf5eDAuBEl1yM"
os.environ["UNSTRUCTURED_API_KEY"] = "uXNQkQUcl6RM1u31i7BCHG53XdGeWU"

from api.index import app

client = TestClient(app)

def test_full_pipeline():
    print("🚀 Running FULL INTEGRATION TEST...")
    
    # 0. Health Check
    print("🔍 Checking API Health...")
    res = client.get("/api")
    print(f"Health Response: {res.json()}")
    
    # 1. Ingest
    pdf_path = "docs/attention-is-all-you-need.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF {pdf_path} not found.")
        return

    print(f"📦 Step 1: Ingesting {pdf_path}...")
    with open(pdf_path, "rb") as f:
        res = client.post("/api/ingest", files={"file": f})
        
    print(f"Ingest Status: {res.status_code}")
    if res.status_code != 200:
        print(f"❌ Ingest Failed! Body: {res.text}")
        return
    
    print(f"✅ Ingest Successful: {res.json()}")

    # 2. Query
    print("🧠 Step 2: Querying...")
    query_data = {"query": "What are the main components of the transformer?"}
    res = client.post("/api/query", data=query_data)
    
    print(f"Query Status: {res.status_code}")
    if res.status_code != 200:
        print(f"❌ Query Failed! Body: {res.text}")
        return

    print("✅ Query Successful!")
    answer = res.json().get("answer", "No answer found")
    print("\n" + "="*50)
    print("AI RESPONSE:")
    print(answer)
    print("="*50 + "\n")

if __name__ == "__main__":
    test_full_pipeline()
