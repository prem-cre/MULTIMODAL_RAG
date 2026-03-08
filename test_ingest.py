import requests
import os

url = "http://localhost:8000/api/ingest"
pdf_path = "docs/attention-is-all-you-need.pdf"

if not os.path.exists(pdf_path):
    print(f"ERROR: {pdf_path} not found. Please put a sample PDF in the root directory.")
else:
    with open(pdf_path, "rb") as f:
        files = {"file": f}
        try:
            res = requests.post(url, files=files)
            print(f"Status: {res.status_code}")
            print(f"Response: {res.text}")
        except Exception as e:
            print(f"Exception: {e}")
