#!/usr/bin/env python3
"""Check tracked/staged paths and new commit subjects using only the stdlib."""

import argparse
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


def git(*args):
    return subprocess.check_output(["git", *args])


def forbidden(path):
    p = PurePosixPath(path)
    if set(p.parts) & {"node_modules", "venv", ".venv", "__pycache__"}:
        return "installed dependencies or Python cache"
    if p.name == ".DS_Store" or p.suffix in {".pyc", ".pyo"}:
        return "machine-generated file"
    if p.name == ".env" or (
        p.name.startswith(".env.")
        and p.suffix not in {".example", ".sample", ".template"}
    ):
        return "local environment file; commit an example instead"
    if ".nira" in p.parts and p.suffix in {".db", ".sqlite", ".sqlite3"}:
        return "local issue-tracker database"
    return None


def subject_error(message):
    subject = message.splitlines()[0].strip() if message.splitlines() else ""
    if not subject or re.match(r"(?i)^(wip\b|fixup!|squash!)", subject):
        return "replace the temporary message with a description of the change"
    if re.fullmatch(
        r"(?i)(update(?:s|d)?(?: site| content| readme)?|fix(?:es|ed)?|changes?|"
        r"test(?:s)?|cleanup|gitignore|first commit|initial commit)[.!]?",
        subject,
    ):
        return "describe what changed (for example: Handle expired peer connections)"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--commit-msg", type=Path)
    parser.add_argument("--range", dest="commit_range")
    args = parser.parse_args()
    errors = []
    if args.commit_msg:
        error = subject_error(args.commit_msg.read_text())
        if error:
            errors.append(error)
    else:
        command = (
            ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
            if args.staged
            else ["ls-files", "-z"]
        )
        for path in git(*command).decode("utf-8", "surrogateescape").split("\0"):
            if path and (reason := forbidden(path)):
                errors.append(f"{path}: {reason}")
        if (
            args.staged
            and subprocess.run(["git", "diff", "--cached", "--check"]).returncode
        ):
            errors.append("fix whitespace errors in the staged diff")
    if args.commit_range:
        for sha in (
            git("rev-list", "--no-merges", args.commit_range).decode().splitlines()
        ):
            message = git("show", "-s", "--format=%B", sha).decode("utf-8", "replace")
            if error := subject_error(message):
                errors.append(f"{sha[:12]}: {error}")
    for error in errors:
        print(error, file=sys.stderr)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
