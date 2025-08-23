#!/usr/bin/env python3
"""Test script to verify backup failover returns 200 status."""

import requests
import json
import sys

# Test with intentionally broken primary endpoint (port 8001 instead of 8003)
# The backup should kick in and return success

test_file_content = """
This is a test document to verify the backup embeddings provider.
When the primary NVIDIA provider fails, the backup Bedrock provider should
take over seamlessly and return a 200 status code to the client.
"""

payload = {
    "file_id": "test-failover-123",
    "user_id": "test-user",
    "pinecone_index_name": "test-index",
    "content": test_file_content,
    "filename": "test_failover.txt"
}

url = "http://localhost:8001/embed"

print("Testing failover with intentionally broken primary provider...")
print(f"Sending request to {url}")

try:
    response = requests.post(url, json=payload, timeout=30)
    
    print(f"\nResponse Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS: Backup failover working correctly - returned 200")
        result = response.json()
        if "message" in result:
            print(f"Message: {result['message']}")
    else:
        print(f"❌ FAILURE: Got status {response.status_code} instead of 200")
        print(f"Response: {response.text}")
        sys.exit(1)
        
except requests.exceptions.Timeout:
    print("❌ Request timed out")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)