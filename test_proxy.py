import urllib.request
import sys

for i in range(10):
    try:
        proxy_handler = urllib.request.ProxyHandler({
            'http': 'http://proxy-rotator:8080',
            'https': 'http://proxy-rotator:8080'
        })
        opener = urllib.request.build_opener(proxy_handler)
        # Test HTTPS to verify CONNECT logic
        url = 'https://api.ipify.org'
        req = urllib.request.Request(url)
        req.add_header('Authorization', f'Bearer test-key-{i}')
        
        with opener.open(req, timeout=10) as response:
            print(f"Attempt {i+1} SUCCESS [{url}]: {response.read().decode().strip()}")
            sys.exit(0)
    except Exception as e:
        print(f"Attempt {i+1} FAILED: {e}")

print("All attempts failed.")
sys.exit(1)
