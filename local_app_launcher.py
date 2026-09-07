from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


APP_NAME = "Slope Safety App"
DEFAULT_HOST = "127.0.0.1"
PORT_START = 8501
PORT_END = 8599


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def app_dir() -> Path:
    root = resource_root()
    candidates = [
        root / "app",
        Path(sys.executable).resolve().parent / "app",
        Path(__file__).resolve().parent / "app",
    ]
    for candidate in candidates:
        if (candidate / "app.py").exists():
            return candidate
    raise FileNotFoundError("Could not find bundled app/app.py.")


def find_free_port(start: int = PORT_START, end: int = PORT_END) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((DEFAULT_HOST, port)) != 0:
                return port
    raise RuntimeError(f"No free localhost port found in {start}-{end}.")


def health_url(port: int) -> str:
    return f"http://{DEFAULT_HOST}:{port}/_stcore/health"


def app_url(port: int) -> str:
    return f"http://localhost:{port}"


def wait_until_ready(port: int, process: subprocess.Popen, timeout_seconds: int = 60) -> bool:
    deadline = time.time() + timeout_seconds
    url = health_url(port)
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status == 200 and response.read().decode("utf-8", errors="ignore").strip() == "ok":
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    return False


def child_command(port: int) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--serve", "--port", str(port)]
    return [sys.executable, str(Path(__file__).resolve()), "--serve", "--port", str(port)]


def launch_parent() -> int:
    port = find_free_port()
    command = child_command(port)
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    print(f"{APP_NAME} is starting on {app_url(port)}")
    print("Keep this window open while using the app.")
    print("Press Ctrl+C or close this window to stop it.")
    print()

    process = subprocess.Popen(command, cwd=str(resource_root()), env=env)
    try:
        if not wait_until_ready(port, process):
            print("The local app did not become ready. Check the messages above for details.")
            return process.poll() or 1

        webbrowser.open(app_url(port), new=2)
        print(f"{APP_NAME} is running at {app_url(port)}")
        while process.poll() is None:
            time.sleep(1)
        return process.returncode or 0
    except KeyboardInterrupt:
        print("Stopping local app...")
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()


def serve_streamlit(port: int) -> int:
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = DEFAULT_HOST
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    import streamlit.config as config
    from streamlit.web import bootstrap

    config.set_option("server.headless", True, "local launcher")
    config.set_option("server.address", DEFAULT_HOST, "local launcher")
    config.set_option("server.port", port, "local launcher")
    config.set_option("server.fileWatcherType", "none", "local launcher")
    config.set_option("browser.gatherUsageStats", False, "local launcher")
    config.set_option("global.developmentMode", False, "local launcher")

    app_path = app_dir() / "app.py"
    flag_options = {
        "server_headless": True,
        "server_address": DEFAULT_HOST,
        "server_port": port,
        "server_fileWatcherType": "none",
        "browser_gatherUsageStats": False,
        "global_developmentMode": False,
    }
    bootstrap.run(str(app_path), False, [], flag_options)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local slope safety Streamlit app.")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=PORT_START, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.serve:
        return serve_streamlit(args.port)
    return launch_parent()


if __name__ == "__main__":
    raise SystemExit(main())
