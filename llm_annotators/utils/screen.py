import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Optional

_SCREEN_CMD_FRAGMENT = "SCREEN -S "
_SCREEN_LINE_RE = re.compile(r"\s*(\d+\.[^\s]+)")


def resolve_screen_pid(session: str) -> Optional[int]:
    """
    Resolve a screen session name to its PID.

    Accepts either full session name (e.g. "2176136.run-foo") or the
    short session name (e.g. "run-foo").
    """
    match = re.match(r"^(?P<pid>\d+)\.", session)
    if match:
        pid = int(match.group("pid"))
        if os.path.exists(f"/proc/{pid}"):
            return pid

    try:
        ps_output = subprocess.check_output(["ps", "-ef"], text=True)
    except (OSError, subprocess.SubprocessError):
        return None

    for line in ps_output.splitlines():
        if _SCREEN_CMD_FRAGMENT not in line:
            continue
        if f"{_SCREEN_CMD_FRAGMENT}{session}" in line or session in line:
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])

    return None


def list_screen_sessions() -> list[str]:
    """
    Return the active screen session names (e.g. "2176136.run-foo").
    """
    try:
        output = subprocess.check_output(["screen", "-ls"], text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Failed to run screen -ls") from exc

    sessions: list[str] = []
    for line in output.splitlines():
        match = _SCREEN_LINE_RE.match(line)
        if match:
            sessions.append(match.group(1))
    return sessions


def list_screen_sessions_matching(substring: str) -> list[str]:
    """
    Return screen session names that contain the given substring.
    """
    return [session for session in list_screen_sessions() if substring in session]


def get_process_env(pid: int) -> dict[str, str]:
    """
    Read environment variables for a process by PID.
    """
    env_path = f"/proc/{pid}/environ"
    with open(env_path, "rb") as env_file:
        raw_env = env_file.read()

    env: dict[str, str] = {}
    for item in raw_env.split(b"\0"):
        if not item:
            continue
        key, _, value = item.partition(b"=")
        env[key.decode()] = value.decode(errors="replace")
    return env


def query_screen_env_var(session: str, var_name: str) -> Optional[str]:
    """
    Query an environment variable directly from a screen session's shell.

    This handles the case where the variable was exported after the shell started
    and thus doesn't appear in /proc/[pid]/environ.

    Note: We use an external script because screen's 'stuff' command interprets
    $identifier specially and strips unrecognized variables.
    """
    script_path = None
    output_path = None

    try:
        # Create a script that will echo the variable
        # We use a script because screen -X stuff interprets $VAR specially
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".sh"
        ) as script_file:
            script_path = script_file.name
            script_file.write(f'printf "%s" "${var_name}"\n')

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".out"
        ) as output_file:
            output_path = output_file.name

        os.chmod(script_path, 0o755)

        # Execute the script in the screen session
        cmd = f"bash {script_path} > {output_path}"
        subprocess.run(
            ["screen", "-S", session, "-p", "0", "-X", "stuff", cmd + "\n"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        # Give the shell a moment to execute the command
        time.sleep(0.15)

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                value = f.read()
            # Return None if the variable was unset (empty output)
            return value if value else None
        return None
    except (subprocess.SubprocessError, OSError, TimeoutError):
        return None
    finally:
        for path in (script_path, output_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def _list_processes() -> list[tuple[int, int, str]]:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,command="], text=True
        )
    except (OSError, subprocess.SubprocessError):
        return []

    processes: list[tuple[int, int, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_str, ppid_str, cmd = parts
        if not (pid_str.isdigit() and ppid_str.isdigit()):
            continue
        processes.append((int(pid_str), int(ppid_str), cmd))
    return processes


def _build_children_map(
    processes: list[tuple[int, int, str]],
) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for pid, ppid, _cmd in processes:
        children.setdefault(ppid, []).append(pid)
    return children


def _find_descendants(root_pid: int, children: dict[int, list[int]]) -> list[int]:
    descendants: list[int] = []
    stack = [root_pid]
    while stack:
        current = stack.pop()
        for child_pid in children.get(current, []):
            descendants.append(child_pid)
            stack.append(child_pid)
    return descendants


def get_screen_env(session: str) -> dict[str, str]:
    """
    Return environment variables for a screen session.
    """
    pid = resolve_screen_pid(session)
    if pid is None:
        raise ValueError(f"Screen session not found: {session}")
    return get_process_env(pid)


def get_screen_env_var(
    session: str,
    var_name: str,
    default: Optional[str] = None,
    *,
    require: bool = False,
) -> Optional[str]:
    """
    Return a single environment variable from a screen session.
    """
    env = get_screen_env(session)
    value = env.get(var_name)
    if value is None:
        if require:
            raise KeyError(f"Env var not found in screen session: {var_name}")
        return default
    return value


def get_screen_env_var_rows(
    var_names: list[str],
    *,
    session_filter: Optional[str] = None,
    missing: str = "(missing)",
) -> list[list[str]]:
    """
    Return rows of (session, var_value) for matching screen sessions.
    """
    if session_filter:
        sessions = list_screen_sessions_matching(session_filter)
    else:
        sessions = list_screen_sessions()

    rows: list[list[str]] = []
    processes = _list_processes()
    children = _build_children_map(processes)
    for session in sessions:
        pid = resolve_screen_pid(session)
        if pid is None:
            rows.append([session] + [missing for _ in var_names])
            continue
        try:
            env = get_screen_env(session)
        except (FileNotFoundError, PermissionError, OSError):
            env = {}
        row_values = [env.get(var_name) for var_name in var_names]
        if any(value is None for value in row_values):
            # Try descendant processes (e.g., running python process)
            for candidate_pid in _find_descendants(pid, children):
                try:
                    candidate_env = get_process_env(candidate_pid)
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                for idx, var_name in enumerate(var_names):
                    if row_values[idx] is None:
                        row_values[idx] = candidate_env.get(var_name)
                if all(value is not None for value in row_values):
                    break
        # NOTE: We intentionally do NOT fall back to query_screen_env_var()
        # (which uses screen -X stuff) because it injects commands into every
        # session that lacks the variable, polluting unrelated sessions.
        row = [session] + [value if value is not None else missing for value in row_values]
        rows.append(row)
    return rows


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """
    Format rows as an ASCII table.
    """
    if not rows:
        return ""

    column_widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows))
        for col_idx in range(len(headers))
    ]
    border = "+-" + "-+-".join("-" * width for width in column_widths) + "-+"

    header_cells = " | ".join(
        f"{headers[col_idx]:<{column_widths[col_idx]}}"
        for col_idx in range(len(headers))
    )
    lines = [border, f"| {header_cells} |", border]
    for row in rows:
        row_cells = " | ".join(
            f"{row[col_idx]:<{column_widths[col_idx]}}"
            for col_idx in range(len(headers))
        )
        lines.append(f"| {row_cells} |")
    lines.append(border)
    return "\n".join(lines)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_env_parser = subparsers.add_parser(
        "show-env-vars",
        help="Show environment variables from screen sessions",
    )
    show_env_parser.add_argument(
        "var",
        nargs="+",
        help="Environment variable name(s) to display",
    )
    show_env_parser.add_argument(
        "--filter",
        default=None,
        help="Only include sessions containing this substring",
    )
    show_env_parser.add_argument(
        "--missing",
        default="(missing)",
        help="Value to show when the env var is missing",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "show-env-vars":
        if args.filter == "":
            args.filter = None
        rows = get_screen_env_var_rows(
            args.var,
            session_filter=args.filter,
            missing=args.missing,
        )
        if not rows:
            print("No matching screen sessions found.")
            return 0
        print(format_table(["session", *args.var], rows))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
