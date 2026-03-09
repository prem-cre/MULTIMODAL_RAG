import os
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations

load_dotenv()

api_key = os.getenv("UNSTRUCTURED_API_KEY")
print(f"DEBUG: Key found: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

client = UnstructuredClient(
    api_key_auth=api_key,
    server_url="https://api.unstructuredapp.io/general/v0/general"
)

# Test with a very small dummy file or just check connection if possible
# Since I don't have a file handy, I'll just check if the client is initialized correctly
# and maybe try a simple partition on a string if allowed, but usually it needs a file.

print("DEBUG: Client initialized.")
try:
    # Try a dummy request to see if it rejects with 401
    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(content=b"test", file_name="test.txt"),
            strategy="fast",
        )
    )
    res = client.general.partition(request=req)
    print("DEBUG: Request successful (unexpected for dummy data).")
except Exception as e:
    print(f"DEBUG: Caught error: {e}")
