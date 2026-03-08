import requests
import os

url = "http://127.0.0.1:8000/api/ingest"
pdf_path = "docs/attention-is-all-you-need.pdf"

if not os.path.exists(pdf_path):
    print(f"ERROR: {pdf_path} not found.")
else:
    with open(pdf_path, "rb") as f:
        files = {"file": f}
        try:
            print(f"Sending request to {url}...")
            res = requests.post(url, files=files, timeout=180)
            print(f"Status: {res.status_code}")
            print(f"Response: {res.text}")
        except Exception as e:
            print(f"Exception: {e}")
