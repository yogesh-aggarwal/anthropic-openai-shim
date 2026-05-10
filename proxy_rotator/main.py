import asyncio
import hashlib
import time
import logging
import aiohttp
import os
import random
import socket
from typing import Dict, List, Optional

# Configuration
# Check both /app (local dev) and / (Docker mount)
PROXY_FILES = ["PROXIES.txt", "/PROXIES.txt"]
PROXY_LIST_URL = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text&protocol=http&timeout=150"
REFRESH_INTERVAL = 300
MAX_RETRIES = 10
INITIAL_SCORE = 100
FAILURE_PENALTY = 50
SUCCESS_REWARD = 10

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("proxy-rotator")

class ProxyStats:
    def __init__(self, url: str, source: str = "url"):
        self.url = url
        self.source = source # "file" or "url"
        self.score = INITIAL_SCORE
        self.avg_latency = 0.0
        self.total_calls = 0
        self.failures = 0

    def record_success(self, latency: float):
        self.total_calls += 1
        self.score = min(200, self.score + SUCCESS_REWARD)
        if self.avg_latency == 0:
            self.avg_latency = latency
        else:
            self.avg_latency = (self.avg_latency * 0.7) + (latency * 0.3)

    def record_failure(self):
        self.total_calls += 1
        self.failures += 1
        self.score = max(0, self.score - FAILURE_PENALTY)

class ProxyManager:
    def __init__(self):
        self.proxies: Dict[str, ProxyStats] = {}
        self.last_fetch = 0

    async def fetch_proxies(self):
        try:
            new_file_urls = []
            new_url_urls = []
            
            # 1. Read from local files (Priority Source)
            for file_path in PROXY_FILES:
                if os.path.exists(file_path):
                    logger.info(f"Loading priority proxies from {file_path}...")
                    with open(file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line: continue
                            parts = line.split(':')
                            if len(parts) == 4:
                                url = f"http://{parts[2]}:{parts[3]}@{parts[0]}:{parts[1]}"
                                new_file_urls.append(url)
                            elif len(parts) == 2:
                                url = f"http://{parts[0]}:{parts[1]}"
                                new_file_urls.append(url)
            
            # 2. Fetch from URL (Fallback Source)
            logger.info("Fetching secondary proxies from URL...")
            async with aiohttp.ClientSession() as session:
                async with session.get(PROXY_LIST_URL, timeout=30) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        for line in text.splitlines():
                            p = line.strip()
                            if p:
                                if not p.startswith("http"):
                                    p = f"http://{p}"
                                new_url_urls.append(p)

            # Update master list
            incoming_urls = {}
            for u in new_file_urls: incoming_urls[u] = "file"
            for u in new_url_urls: 
                if u not in incoming_urls: incoming_urls[u] = "url"

            if incoming_urls:
                # Preservation logic
                for url, source in incoming_urls.items():
                    if url not in self.proxies:
                        self.proxies[url] = ProxyStats(url, source)
                    else:
                        # Update source in case it moved from URL to File
                        self.proxies[url].source = source
                
                # Cleanup dead ones
                for url in list(self.proxies.keys()):
                    if url not in incoming_urls:
                        del self.proxies[url]
                
                self.last_fetch = time.time()
                file_count = sum(1 for p in self.proxies.values() if p.source == "file")
                url_count = sum(1 for p in self.proxies.values() if p.source == "url")
                logger.info(f"Proxy Pool Updated: {file_count} Priority (File), {url_count} Secondary (URL)")
            else:
                logger.warning("No proxies found from any source!")
        except Exception as e:
            logger.error(f"Error fetching proxies: {e}")

    def get_proxy(self, auth_header: str, attempt: int = 0) -> str:
        if not self.proxies:
            return None
            
        # Priority Logic:
        # 1. Source (File > URL)
        # 2. Score (High > Low)
        # 3. Latency (Low > High)
        sorted_proxies = sorted(
            self.proxies.values(),
            key=lambda p: (
                p.source == "file", # True (1) > False (0)
                p.score, 
                -p.avg_latency if p.avg_latency > 0 else -999
            ),
            reverse=True
        )
        
        # On early attempts, respect the sorting strictly
        if attempt < 3:
            return sorted_proxies[attempt % len(sorted_proxies)].url
        else:
            # On retries, pick from the top 30% randomly to ensure coverage
            sample_size = max(1, int(len(sorted_proxies) * 0.3))
            return random.choice(sorted_proxies[:sample_size]).url

    def report_outcome(self, url: str, success: bool, latency: float = 0):
        if url in self.proxies:
            if success:
                self.proxies[url].record_success(latency)
            else:
                self.proxies[url].record_failure()

proxy_manager = ProxyManager()

async def refresh_loop():
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        await proxy_manager.fetch_proxies()

async def relay_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        while True:
            data = await reader.read(16384)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

async def handle_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        # Read the CONNECT request header
        header_data = b""
        while True:
            line = await reader.readline()
            if not line or line == b"\r\n":
                break
            header_data += line
        
        if not header_data:
            writer.close()
            return

        header_str = header_data.decode("utf-8")
        lines = header_str.split("\r\n")
        if not lines or not lines[0].startswith("CONNECT"):
            writer.close()
            return

        parts = lines[0].split(" ")
        if len(parts) < 2:
            writer.close()
            return
            
        target_addr = parts[1] # e.g. "openrouter.ai:443"
        host, port = target_addr.split(":")
        port = int(port)

        # Get a proxy to use
        # For simplicity in this TCP relay, we don't have the full headers here for user targeting
        proxy_url = proxy_manager.get_proxy("", 0)
        if not proxy_url:
            writer.write(b"HTTP/1.1 503 No Proxies Available\r\n\r\n")
            await writer.drain()
            writer.close()
            return

        logger.info(f"CONNECT {target_addr} via {proxy_url}")

        # Connect to the external proxy and send our CONNECT request to it
        from urllib.parse import urlparse
        p = urlparse(proxy_url)
        
        try:
            proxy_reader, proxy_writer = await asyncio.open_connection(p.hostname, p.port)
            
            # If the outgoing proxy has auth
            auth_header = ""
            if p.username and p.password:
                import base64
                auth = base64.b64encode(f"{p.username}:{p.password}".encode()).decode()
                auth_header = f"Proxy-Authorization: Basic {auth}\r\n"

            proxy_writer.write(f"CONNECT {target_addr} HTTP/1.1\r\n{auth_header}\r\n".encode())
            await proxy_writer.drain()

            # Read response from proxy
            resp = await proxy_reader.readline()
            if b"200" not in resp:
                logger.warning(f"Upstream proxy {proxy_url} rejected CONNECT: {resp.decode().strip()}")
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await writer.drain()
                proxy_writer.close()
                writer.close()
                return

            # Skip remaining upstream headers
            while True:
                line = await proxy_reader.readline()
                if line == b"\r\n":
                    break

            # Signal success to the client
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()

            # Relay bidirectional traffic
            await asyncio.gather(
                relay_stream(reader, proxy_writer),
                relay_stream(proxy_reader, writer)
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to upstream proxy {proxy_url}: {e}")
            writer.write(b"HTTP/1.1 504 Gateway Timeout\r\n\r\n")
            await writer.drain()
            writer.close()

    except Exception as e:
        logger.error(f"Error in handle_connect: {e}")
        try:
            writer.close()
        except:
            pass

async def handle_request_http(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    # Support for regular GET/POST/etc if LiteLLM sends them (non-CONNECT)
    # For now, let's focus on CONNECT as it's what we see in logs.
    # To keep it simple, we'll just log and close if it's not CONNECT
    data = await reader.read(8192)
    if not data:
        writer.close()
        return

    if data.startswith(b"CONNECT"):
        # Put data back or just parse it here?
        # aiohttp is better for HTTP, but we need TCP for CONNECT.
        # Let's simplify: this proxy will ONLY support CONNECT for now.
        pass

async def main():
    await proxy_manager.fetch_proxies()
    asyncio.create_task(refresh_loop())

    server = await asyncio.start_server(handle_connect, "0.0.0.0", 8080)
    logger.info("TCP-based Proxy Rotator running on port 8080 (Supports CONNECT)")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
