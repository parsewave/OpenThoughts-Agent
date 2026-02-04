#!/usr/bin/env python3
"""
Simple HTTP proxy that forwards requests through a SOCKS5 proxy.

This allows HTTP clients (like aiohttp with trust_env=True) to use a SOCKS proxy
by setting HTTP_PROXY/HTTPS_PROXY to this local HTTP proxy.

Usage:
    # Start the proxy (runs in background):
    python http_proxy.py --socks-host 10.128.24.25 --socks-port 21004 &

    # Set environment variables:
    export HTTP_PROXY=http://127.0.0.1:8080
    export HTTPS_PROXY=http://127.0.0.1:8080

    # Now any HTTP client using HTTP_PROXY will route through SOCKS

Requirements:
    pip install aiohttp aiohttp-socks

Based on: https://github.com/romis2012/aiohttp-socks
"""

import argparse
import asyncio
import logging
import signal
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [http-proxy] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class SocksForwardingProxy:
    """HTTP proxy server that forwards requests through a SOCKS5 proxy."""

    def __init__(self, socks_host: str, socks_port: int, listen_port: int = 8080):
        self.socks_host = socks_host
        self.socks_port = socks_port
        self.listen_port = listen_port
        self.server: Optional[asyncio.Server] = None

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Handle an incoming HTTP proxy request."""
        try:
            # Read the request line
            request_line = await reader.readline()
            if not request_line:
                return

            request_line = request_line.decode("utf-8", errors="ignore").strip()
            parts = request_line.split()
            if len(parts) < 3:
                writer.close()
                return

            method, url, _ = parts[0], parts[1], parts[2]

            # Read headers
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                if b":" in line:
                    key, value = line.decode("utf-8", errors="ignore").split(":", 1)
                    headers[key.strip().lower()] = value.strip()

            # Handle CONNECT method (HTTPS tunneling)
            if method == "CONNECT":
                await self._handle_connect(url, reader, writer)
            else:
                await self._handle_request(method, url, headers, reader, writer)

        except Exception as e:
            logger.debug(f"Error handling client: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_connect(
        self, url: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Handle CONNECT method for HTTPS tunneling."""
        try:
            from aiohttp_socks import ProxyConnector, ProxyType

            # Parse host:port from CONNECT request
            if ":" in url:
                host, port = url.rsplit(":", 1)
                port = int(port)
            else:
                host, port = url, 443

            # Connect through SOCKS proxy
            connector = ProxyConnector(
                proxy_type=ProxyType.SOCKS5,
                host=self.socks_host,
                port=self.socks_port,
                rdns=True,  # Remote DNS resolution
            )

            # Get the underlying socket connection through SOCKS
            import aiohttp

            async with aiohttp.ClientSession(connector=connector) as session:
                # Create a raw socket connection through the proxy
                conn = await connector.connect(
                    aiohttp.ClientRequest(
                        "GET",
                        aiohttp.client.URL(f"https://{host}:{port}/"),
                        headers={},
                        loop=asyncio.get_event_loop(),
                    ),
                    [],
                    aiohttp.ClientTimeout(total=30),
                )

                # Send success response
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()

                # Tunnel data bidirectionally
                remote_reader = conn.protocol.reader
                remote_writer = conn.protocol.writer

                await asyncio.gather(
                    self._pipe(reader, remote_writer),
                    self._pipe(remote_reader, writer),
                    return_exceptions=True,
                )

        except Exception as e:
            logger.debug(f"CONNECT error for {url}: {e}")
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()

    async def _handle_request(
        self,
        method: str,
        url: str,
        headers: dict,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle regular HTTP request."""
        try:
            from aiohttp_socks import ProxyConnector, ProxyType
            import aiohttp

            connector = ProxyConnector(
                proxy_type=ProxyType.SOCKS5,
                host=self.socks_host,
                port=self.socks_port,
                rdns=True,
            )

            # Read body if present
            body = None
            if "content-length" in headers:
                length = int(headers["content-length"])
                body = await reader.read(length)

            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.request(
                    method, url, headers=headers, data=body
                ) as response:
                    # Send response status
                    status_line = f"HTTP/1.1 {response.status} {response.reason}\r\n"
                    writer.write(status_line.encode())

                    # Send headers
                    for key, value in response.headers.items():
                        if key.lower() not in ("transfer-encoding", "connection"):
                            writer.write(f"{key}: {value}\r\n".encode())
                    writer.write(b"\r\n")

                    # Send body
                    async for chunk in response.content.iter_chunked(8192):
                        writer.write(chunk)
                        await writer.drain()

        except Exception as e:
            logger.debug(f"Request error for {url}: {e}")
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()

    async def _pipe(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Pipe data from reader to writer."""
        try:
            while True:
                data = await reader.read(8192)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except Exception:
            pass

    async def start(self):
        """Start the proxy server."""
        self.server = await asyncio.start_server(
            self.handle_client, "127.0.0.1", self.listen_port
        )
        logger.info(f"HTTP proxy listening on 127.0.0.1:{self.listen_port}")
        logger.info(f"Forwarding to SOCKS5 at {self.socks_host}:{self.socks_port}")
        logger.info("")
        logger.info("Set these environment variables:")
        logger.info(f"  export HTTP_PROXY=http://127.0.0.1:{self.listen_port}")
        logger.info(f"  export HTTPS_PROXY=http://127.0.0.1:{self.listen_port}")

    async def stop(self):
        """Stop the proxy server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Proxy stopped")


async def main():
    parser = argparse.ArgumentParser(description="HTTP-to-SOCKS proxy")
    parser.add_argument("--socks-host", required=True, help="SOCKS5 proxy host")
    parser.add_argument("--socks-port", type=int, required=True, help="SOCKS5 proxy port")
    parser.add_argument("--listen-port", type=int, default=8080, help="Local HTTP proxy port")
    args = parser.parse_args()

    proxy = SocksForwardingProxy(args.socks_host, args.socks_port, args.listen_port)
    await proxy.start()

    # Handle shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def shutdown():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    await stop_event.wait()
    await proxy.stop()


if __name__ == "__main__":
    asyncio.run(main())
