import requests

url = "http://127.0.0.1:8000/api/query"
data = {"query": "Summarize the Attention mechanism."}

try:
    print(f"Sending query to {url}...")
    res = requests.post(url, data=data, timeout=300)
    print(f"Status: {res.status_code}")
    print(f"Response: {res.text}")
except Exception as e:
    print(f"Exception: {e}")
