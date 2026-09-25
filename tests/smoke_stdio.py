"""Out-of-process smoke check: speak real JSON-RPC to the MCP server over stdio.

Not part of the pytest suite (it spawns a subprocess and needs the `mcp` extra);
run it by hand to confirm the transport itself works:

    python tests/smoke_stdio.py
"""
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    env = {**os.environ, "FAULTSPOOL_DB": str(ROOT / "examples" / ".demo" / "faultspool.db"),
           "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen([sys.executable, "-m", "faultspool.mcp_server"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=str(ROOT), env=env, text=True, encoding="utf-8", bufsize=1)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(ln) for ln in proc.stdout], daemon=True).start()

    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "faultspool-smoke", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "spool_stats", "arguments": {}}},
    ]
    # stdin stays open: the server stops on EOF, cancelling anything still in flight.
    for m in msgs:
        proc.stdin.write(json.dumps(m) + "\n")
        proc.stdin.flush()

    replies = {}
    while len(replies) < 3:
        try:
            line = lines.get(timeout=30).strip()
        except queue.Empty:
            break
        if line.startswith("{"):
            msg = json.loads(line)
            if "id" in msg:
                replies[msg["id"]] = msg
    proc.stdin.close()
    err = proc.stderr.read()
    proc.wait(timeout=30)
    out = json.dumps({k: "<captured>" for k in replies})

    ok = True
    init = replies.get(1, {}).get("result", {})
    print("initialize ->", init.get("serverInfo"), init.get("protocolVersion"))
    ok &= (init.get("serverInfo", {}).get("name") == "faultspool")

    tools = [t["name"] for t in replies.get(2, {}).get("result", {}).get("tools", [])]
    print(f"tools/list -> {len(tools)}: {', '.join(sorted(tools))}")
    ok &= len(tools) == 9

    call = replies.get(3, {}).get("result", {})
    text = (call.get("content") or [{}])[0].get("text", "")
    print("tools/call spool_stats ->", text.replace("\n", " | ")[:160])
    ok &= bool(text) and not call.get("isError")

    if not ok:
        print("FAILED\nstdout:", out[:2000], "\nstderr:", err[:2000])
    else:
        print("\nstdio transport OK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
