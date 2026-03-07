from rag_pipeline import get_api_key

def test_api_key():
    api_key = get_api_key()
    assert api_key is not None, "API Key is missing. Ensure GEMINI_API_KEY is set."
    print("✅ GEMINI_API_KEY is properly configured.")

if __name__ == "__main__":
    test_api_key()
    print("Tests completed.")
