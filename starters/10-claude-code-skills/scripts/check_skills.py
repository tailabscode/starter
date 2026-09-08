#!/usr/bin/env python3
"""Validate every skills/<name>/SKILL.md in this starter.

This is deliberately not a pytest suite and does not depend on a real YAML
parser -- the frontmatter these skills use is two flat scalar fields
(name, description), so a stdlib-only split-on-'---' parser is enough and
keeps this starter dependency-free. See the "Tests" section of this
starter's README.

Checks, per SKILL.md:
  1. The file starts with a '---' delimited frontmatter block.
  2. The frontmatter parses into key: value pairs with no blank keys.
  3. 'name' is present, non-empty, and matches the skill's directory name
     (the real Claude Code skill loader keys off the directory name, so a
     mismatch here is a real bug, not a style nit).
  4. 'description' is present and non-empty.
  5. The instructions body (everything after the closing '---') is
     non-empty and long enough to plausibly be real instructions, not a
     stub.

Usage:
    python3 scripts/check_skills.py
Exit code 0 if every skill passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

MIN_BODY_CHARS = 200  # a real instruction set, not a one-line stub

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


class SkillError(Exception):
    """A validation failure for one SKILL.md, with a human-readable reason."""


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a SKILL.md into its (frontmatter dict, body) parts.

    Expects the file to start with '---', a block of 'key: value' lines,
    a closing '---', then the markdown body. Raises SkillError if the
    delimiters are missing or a frontmatter line isn't 'key: value'.
    """
    if not text.startswith("---"):
        raise SkillError("file must start with '---' frontmatter delimiter")

    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillError("expected a closing '---' after the frontmatter block")

    _, frontmatter_block, body = parts

    frontmatter: dict[str, str] = {}
    for line_no, raw_line in enumerate(frontmatter_block.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise SkillError(f"frontmatter line {line_no} is not 'key: value': {raw_line!r}")
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            raise SkillError(f"frontmatter line {line_no} has an empty key: {raw_line!r}")
        frontmatter[key] = value

    return frontmatter, body


def check_skill(skill_dir: Path) -> list[str]:
    """Return a list of problems for one skill directory. Empty list = pass."""
    problems: list[str] = []
    skill_file = skill_dir / "SKILL.md"

    if not skill_file.is_file():
        return [f"missing {skill_file.name}"]

    text = skill_file.read_text(encoding="utf-8")

    try:
        frontmatter, body = parse_frontmatter(text)
    except SkillError as exc:
        return [str(exc)]

    name = frontmatter.get("name", "")
    if not name:
        problems.append("frontmatter 'name' is missing or empty")
    elif name != skill_dir.name:
        problems.append(f"frontmatter name {name!r} does not match directory {skill_dir.name!r}")

    description = frontmatter.get("description", "")
    if not description:
        problems.append("frontmatter 'description' is missing or empty")

    body_stripped = body.strip()
    if not body_stripped:
        problems.append("instructions body is empty")
    elif len(body_stripped) < MIN_BODY_CHARS:
        problems.append(
            f"instructions body is only {len(body_stripped)} chars "
            f"(expected at least {MIN_BODY_CHARS}) -- looks like a stub"
        )

    return problems


def main() -> int:
    if not SKILLS_DIR.is_dir():
        print(f"No skills/ directory found at {SKILLS_DIR}", file=sys.stderr)
        return 1

    skill_dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
    if not skill_dirs:
        print(f"No skill directories found under {SKILLS_DIR}", file=sys.stderr)
        return 1

    all_ok = True
    for skill_dir in skill_dirs:
        problems = check_skill(skill_dir)
        if problems:
            all_ok = False
            print(f"FAIL  {skill_dir.name}")
            for problem in problems:
                print(f"        - {problem}")
        else:
            print(f"PASS  {skill_dir.name}")

    print()
    print("All skills valid." if all_ok else "One or more skills failed validation.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
