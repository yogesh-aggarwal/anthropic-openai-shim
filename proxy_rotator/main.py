import asyncio
import hashlib
import time
import logging
import aiohttp

# Configuration
PROXY_LIST_URL = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text&protocol=http&timeout=150"
REFRESH_INTERVAL = 300
LISTEN_PORT = 8080

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("proxy-rotator")

class ProxyManager:
    def __init__(self):
        self.proxies = []
        self.last_fetch = 0

    async def fetch_proxies(self):
        try:
            logger.info("Fetching fresh proxy list...")
            async with aiohttp.ClientSession() as session:
                async with session.get(PROXY_LIST_URL, timeout=30) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to fetch proxies: {resp.status}")
                        return
                    text = await resp.text()
                    new_proxies = []
                    for line in text.splitlines():
                        p = line.strip()
                        if p:
                            if not p.startswith("http"):
                                p = f"http://{p}"
                            new_proxies.append(p)
                    
                    if new_proxies:
                        self.proxies = new_proxies
                        self.last_fetch = time.time()
                        logger.info(f"Successfully loaded {len(self.proxies)} proxies")
                    else:
                        logger.warning("Fetched proxy list is empty")
        except Exception as e:
            logger.error(f"Error fetching proxies: {e}")

    def get_proxy(self, auth_header: str) -> str:
        if not self.proxies:
            return None
        h = hashlib.sha256(auth_header.encode() if auth_header else b"anonymous").hexdigest()
        idx = int(h, 16) % len(self.proxies)
        return self.proxies[idx]

proxy_manager = ProxyManager()

async def pipe(reader, writer):
    try:
        while not reader.at_eof():
            data = await reader.read(16384)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()

async def handle_client(reader, writer):
    try:
        # Peek at headers to find Authorization
        data = await reader.read(16384)
        if not data:
            writer.close()
            return

        header_text = data.decode(errors='ignore')
        method = "UNKNOWN"
        target_url = "unknown"
        auth_header = ""
        lines = header_text.split('\r\n')
        if lines:
            parts = lines[0].split()
            if len(parts) > 1:
                method = parts[0].upper()
                target_url = parts[1]

        for line in lines:
            if line.lower().startswith('authorization:'):
                auth_header = line.split(':', 1)[1].strip()
                break

        proxy_url = proxy_manager.get_proxy(auth_header or target_url)
        if not proxy_url:
            logger.warning("No proxies available to handle request")
            writer.write(b"HTTP/1.1 503 Service Unavailable\r\n\r\n")
            await writer.drain()
            writer.close()
            return

        # Parse proxy
        p_parts = proxy_url.split('//')[-1].split(':')
        p_host = p_parts[0]
        p_port = int(p_parts[1])

        logger.info(f"Proxying {method} {target_url} -> {p_host}:{p_port} [Auth: {auth_header[:8]}...]")

        # Connect to upstream proxy
        try:
            remote_reader, remote_writer = await asyncio.open_connection(p_host, p_port)
        except Exception as e:
            logger.error(f"Failed to connect to upstream proxy {p_host}:{p_port}: {e}")
            if method == 'CONNECT':
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            else:
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\n\r\nBad Gateway")
            await writer.drain()
            writer.close()
            return

        # Forward initial chunk
        remote_writer.write(data)
        await remote_writer.drain()

        # Bidirectional tunnel
        asyncio.create_task(pipe(reader, remote_writer))
        await pipe(remote_reader, writer)

    except Exception as e:
        logger.error(f"Handler error: {e}")
    finally:
        writer.close()

async def refresh_loop():
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        await proxy_manager.fetch_proxies()

async def main():
    await proxy_manager.fetch_proxies()
    asyncio.create_task(refresh_loop())
    
    server = await asyncio.start_server(handle_client, '0.0.0.0', 8080)
    addr = server.sockets[0].getsockname()
    logger.info(f"Serving on {addr}")

    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
