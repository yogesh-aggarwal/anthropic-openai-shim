import time
import urllib.request

for i in range(10):
    try:
        # Use the GATEWAY URL instead of a ProxyHandler
        url = f"http://proxy-rotator:8080/openrouter/chat/completions"

        req = urllib.request.Request(url, method="POST")
        req.add_header("Authorization", f"Bearer test-key-{i}")
        req.add_header("Content-Type", "application/json")

        # Dummy body
        data = b'{"model": "test", "messages": [{"role": "user", "content": "hi"}]}'

        start = time.time()
        with urllib.request.urlopen(req, data=data, timeout=10) as response:
            latency = time.time() - start
            print(f"Attempt {i + 1} SUCCESS: {response.status} in {latency:.2f}s")
            # We don't exit here so we can see multiple attempts and scoring
    except Exception as e:
        print(f"Attempt {i + 1} FAILED: {e}")

print("Test loop finished.")
