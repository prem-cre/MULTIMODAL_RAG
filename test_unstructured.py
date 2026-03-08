import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_unstructured_fixed():
    print("Testing Unstructured API with the fixed URL...")
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    # Note: the host in the code is https://api.unstructuredapp.io/general/v0/general
    url = "https://api.unstructuredapp.io/general/v0/general"
    headers = {"unstructured-api-key": api_key}
    
    try:
        # Send a minimal valid request or just check if we get a 401
        res = requests.post(url, headers=headers)
        if res.status_code == 401:
            print(f"❌ Unstructured API key is INVALID. Status: {res.status_code}")
        elif res.status_code == 400:
            print(f"✅ Unstructured API key is VALID. (Received 400 for empty request as expected)")
        else:
            print(f"ℹ️ Unstructured API returned: {res.status_code}")
            if res.status_code == 200:
                print("✅ Unstructured API is working perfectly!")
    except Exception as e:
        print(f"❌ Unstructured API connection error: {e}")

if __name__ == "__main__":
    test_unstructured_fixed()
