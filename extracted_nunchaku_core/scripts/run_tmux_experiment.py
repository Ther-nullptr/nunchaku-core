#!/usr/bin/env python3
"""Launch and inspect long-running Nunchaku experiments in tmux."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "runs" / "tmux"
DEFAULT_PREFIX = "nunchaku"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._-") or "experiment"


def _tmux_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("tmux")
    if found is None:
        raise SystemExit(
            "tmux not found in PATH. Run with the triton env, e.g. "
            "`conda run -n triton python scripts/run_tmux_experiment.py ...`."
        )
    return found


def _session_name(prefix: str, name: str) -> str:
    prefix_clean = _sanitize_name(prefix)
    name_clean = _sanitize_name(name)
    if name_clean.startswith(prefix_clean + "_"):
        return name_clean
    return f"{prefix_clean}_{name_clean}"


def _strip_remainder(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing experiment command after `--`")
    return command


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _session_exists(tmux: str, session: str) -> bool:
    return subprocess.run([tmux, "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _latest_meta(log_dir: Path, session: str) -> Path | None:
    matches = sorted(log_dir.glob(f"*_{session}.json"))
    return matches[-1] if matches else None


def start(args: argparse.Namespace) -> int:
    tmux = _tmux_bin(args.tmux_bin)
    command = _strip_remainder(args.command)
    session = _session_name(args.prefix, args.name)
    if _session_exists(tmux, session) and not args.replace:
        raise SystemExit(f"tmux session already exists: {session}. Use --replace to stop and restart it.")
    if _session_exists(tmux, session) and args.replace:
        subprocess.run([tmux, "kill-session", "-t", session], check=True)

    cwd = Path(args.cwd).expanduser().resolve()
    if not cwd.exists():
        raise SystemExit(f"cwd does not exist: {cwd}")
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = _timestamp()
    log_path = log_dir / f"{stamp}_{session}.log"
    status_path = log_dir / f"{stamp}_{session}.status"
    meta_path = log_dir / f"{stamp}_{session}.json"

    command_shell = shlex.join(command)
    if args.conda_env:
        run_shell = shlex.join(["conda", "run", "-n", args.conda_env, *command])
    else:
        run_shell = command_shell

    shell_script = f"""
set -o pipefail
(
  cd {shlex.quote(str(cwd))}
  echo "[tmux-run] session={session}"
  echo "[tmux-run] cwd={cwd}"
  echo "[tmux-run] command={command_shell}"
  echo "[tmux-run] started=$(date -Is)"
  set +e
  {run_shell}
  status=$?
  set -e
  echo "[tmux-run] finished=$(date -Is) status=${{status}}"
  printf "%s\\n" "${{status}}" > {shlex.quote(str(status_path))}
  exit "${{status}}"
) 2>&1 | tee -a {shlex.quote(str(log_path))}
exit "${{PIPESTATUS[0]}}"
""".strip()

    metadata = {
        "session": session,
        "name": args.name,
        "prefix": args.prefix,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(cwd),
        "command": command,
        "command_shell": command_shell,
        "conda_env": args.conda_env,
        "log_path": str(log_path),
        "status_path": str(status_path),
        "tmux_bin": tmux,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    subprocess.run([tmux, "new-session", "-d", "-s", session, "bash", "-lc", shell_script], check=True)
    print(f"started {session}")
    print(f"log: {log_path}")
    print(f"meta: {meta_path}")
    print(f"attach: {tmux} attach -t {session}")
    return 0


def list_sessions(args: argparse.Namespace) -> int:
    tmux = _tmux_bin(args.tmux_bin)
    proc = _run([tmux, "list-sessions", "-F", "#{session_name}\t#{session_created}\t#{session_attached}"], check=False)
    if proc.returncode != 0:
        if "no server running" in proc.stderr:
            return 0
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        return proc.returncode
    prefix = _sanitize_name(args.prefix) + "_"
    for line in proc.stdout.splitlines():
        if not args.all and not line.startswith(prefix):
            continue
        print(line)
    return 0


def status(args: argparse.Namespace) -> int:
    tmux = _tmux_bin(args.tmux_bin)
    session = _session_name(args.prefix, args.name)
    log_dir = Path(args.log_dir).expanduser().resolve()
    meta = _latest_meta(log_dir, session)
    running = _session_exists(tmux, session)
    print(f"session: {session}")
    print(f"running: {str(running).lower()}")
    if meta is None:
        print(f"metadata: not found under {log_dir}")
        return 0 if running else 1
    data = json.loads(meta.read_text(encoding="utf-8"))
    print(f"metadata: {meta}")
    print(f"log: {data['log_path']}")
    status_path = Path(data["status_path"])
    if status_path.exists():
        print(f"exit_status: {status_path.read_text(encoding='utf-8').strip()}")
    else:
        print("exit_status: pending")
    return 0


def tail(args: argparse.Namespace) -> int:
    session = _session_name(args.prefix, args.name)
    log_dir = Path(args.log_dir).expanduser().resolve()
    meta = _latest_meta(log_dir, session)
    if meta is None:
        raise SystemExit(f"metadata not found for session {session} under {log_dir}")
    data = json.loads(meta.read_text(encoding="utf-8"))
    log_path = Path(data["log_path"])
    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")
    cmd = ["tail", "-n", str(args.lines)]
    if args.follow:
        cmd.append("-f")
    cmd.append(str(log_path))
    return subprocess.call(cmd)


def attach(args: argparse.Namespace) -> int:
    tmux = _tmux_bin(args.tmux_bin)
    session = _session_name(args.prefix, args.name)
    return subprocess.call([tmux, "attach", "-t", session])


def stop(args: argparse.Namespace) -> int:
    tmux = _tmux_bin(args.tmux_bin)
    session = _session_name(args.prefix, args.name)
    if not _session_exists(tmux, session):
        print(f"session not running: {session}")
        return 1
    subprocess.run([tmux, "kill-session", "-t", session], check=True)
    print(f"stopped {session}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmux-bin", default=None)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    sub = parser.add_subparsers(dest="cmd", required=True)

    start_p = sub.add_parser("start", help="start a detached tmux experiment")
    start_p.add_argument("name")
    start_p.add_argument("--cwd", default=str(ROOT))
    start_p.add_argument("--conda-env", default=os.environ.get("NUNCHAKU_TMUX_CONDA_ENV", "triton"))
    start_p.add_argument("--replace", action="store_true")
    start_p.add_argument("command", nargs=argparse.REMAINDER)
    start_p.set_defaults(func=start)

    list_p = sub.add_parser("ls", help="list tmux sessions")
    list_p.add_argument("--all", action="store_true")
    list_p.set_defaults(func=list_sessions)

    status_p = sub.add_parser("status", help="show latest metadata and exit status")
    status_p.add_argument("name")
    status_p.set_defaults(func=status)

    tail_p = sub.add_parser("tail", help="tail the latest log for a session")
    tail_p.add_argument("name")
    tail_p.add_argument("-n", "--lines", type=int, default=80)
    tail_p.add_argument("-f", "--follow", action="store_true")
    tail_p.set_defaults(func=tail)

    attach_p = sub.add_parser("attach", help="attach to a running session")
    attach_p.add_argument("name")
    attach_p.set_defaults(func=attach)

    stop_p = sub.add_parser("stop", help="kill a running session")
    stop_p.add_argument("name")
    stop_p.set_defaults(func=stop)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
