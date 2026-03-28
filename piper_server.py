#!/usr/bin/env python3
"""Persistent Piper TTS HTTP server — keeps model loaded in memory to avoid reload delays."""

import subprocess
import threading
import select
import json
import os
import sys
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler

HOST = "127.0.0.1"
PORT = 5111

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_BIN = os.path.join(SCRIPT_DIR, "piper", "piper")
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "piper", "en_US-bmo_voice.onnx")


class PiperProcess:
    """Maintains a persistent Piper subprocess with the model loaded in memory."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.lock = threading.Lock()
        self.process = None
        self._start()

    def _start(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        self.process = subprocess.Popen(
            [PIPER_BIN, "--model", self.model_path, "--output-raw", "--quiet"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # Make stdout non-blocking for select-based reading
        os.set_blocking(self.process.stdout.fileno(), False)
        print(f"[piper_server] Piper process started (pid={self.process.pid})", flush=True)

    def synthesize(self, text):
        with self.lock:
            if self.process.poll() is not None:
                print("[piper_server] Piper process died, restarting...", flush=True)
                self._start()

            self.process.stdin.write((text + "\n").encode())
            self.process.stdin.flush()

            chunks = []
            timeout = 10.0  # Initial timeout — allow time for phonemization
            while True:
                ready, _, _ = select.select([self.process.stdout], [], [], timeout)
                if ready:
                    data = self.process.stdout.read(65536)
                    if data:
                        chunks.append(data)
                        timeout = 0.5  # After first data arrives, use shorter timeout
                    else:
                        # read() returned empty on a non-blocking fd — no data yet
                        if chunks:
                            break
                        timeout = 0.5
                else:
                    break  # Timeout — utterance complete (or nothing generated)

            return b"".join(chunks)

    def shutdown(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()


piper = None


class TTSHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            text = data.get("text", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            text = body.decode(errors="replace")

        if not text.strip():
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"No text provided")
            return

        audio = piper.synthesize(text)

        self.send_response(200)
        self.send_header("Content-Type", "audio/raw")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def do_GET(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass  # Suppress per-request logging


def main():
    global piper

    # Resolve model path from config or default
    model_path = DEFAULT_MODEL
    config_path = os.path.join(SCRIPT_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            vm = cfg.get("voice_model")
            if vm:
                resolved = os.path.join(SCRIPT_DIR, vm) if not os.path.isabs(vm) else vm
                if os.path.exists(resolved):
                    model_path = resolved
        except Exception:
            pass

    print(f"[piper_server] Loading model: {model_path}", flush=True)
    piper = PiperProcess(model_path)

    server = HTTPServer((HOST, PORT), TTSHandler)
    print(f"[piper_server] Listening on http://{HOST}:{PORT}", flush=True)

    def shutdown_handler(sig, frame):
        print("\n[piper_server] Shutting down...", flush=True)
        piper.shutdown()
        server.shutdown()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        server.serve_forever()
    finally:
        piper.shutdown()


if __name__ == "__main__":
    main()
