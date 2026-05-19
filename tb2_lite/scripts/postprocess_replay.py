#!/usr/bin/env python3
"""Post-process incremental replay results — JSON repair only.

Strategy:
  1. For invalid JSON: try to repair truncated JSON by closing brackets
  2. If repaired JSON has commands: use them (already extracted by original parse_prediction)
  3. For empty pred_command_units with invalid JSON: try keystroke extraction on pred_preview
  4. NEVER extract commands from analysis text — those are mentions, not predictions

Key insight: pred_preview is 500 chars but the full prediction can be ~3000 chars.
The original parse_prediction already ran on the FULL prediction, so pred_command_units
is already the best extraction from the full text. Post-processing can only help when:
  - JSON is repairable (rare with 500 chars)
  - The 500-char preview has keystrokes that the full prediction somehow missed (very rare)

Usage:
  python tb2_lite/scripts/postprocess_replay.py <incr.jsonl> [output.json]
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# ── Core metrics (inline to avoid import issues) ─────────────────────────────

def tokenize_command(command: str) -> list[str]:
    normalized = command.strip().lower()
    if not normalized:
        return []
    if normalized == "<wait>":
        return ["<wait>"]
    try:
        return shlex.split(normalized)
    except Exception:
        return normalized.split()


def token_f1(left: str, right: str) -> float:
    left_tokens = tokenize_command(left)
    right_tokens = tokenize_command(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def score_commands(pred_units: list[str], ref_units: list[str]) -> tuple[bool, float, float, float]:
    if not pred_units or not ref_units:
        first_exact = pred_units == ref_units
        return first_exact, 0.0, 0.0, 0.0 if (pred_units or ref_units) else 1.0
    first_exact = pred_units[0] == ref_units[0]
    n = max(len(pred_units), len(ref_units))
    f1s = [token_f1(p, r) for p, r in zip(pred_units, ref_units)]
    return first_exact, sum(f1s) / n, sum(f1s) / n, sum(f1s) / n


def normalize_units(units: list[str]) -> list[str]:
    return [u.strip() for u in units if u.strip()]


# ── JSON repair only ─────────────────────────────────────────────────────────

def repair_and_extract(text: str, existing_units: list[str]) -> tuple[bool, list[str], bool, bool, str]:
    """Conservative: JSON repair + keystroke regex only. No text-based extraction.

    Returns (valid_json, commands, has_analysis, has_plan, note).
    """
    has_analysis = '"analysis"' in text
    has_plan = '"plan"' in text

    # Strategy 1: Repair truncated JSON
    start = text.find("{")
    if start < 0:
        return False, existing_units, has_analysis, has_plan, "no_json"

    blob = text[start:]

    # Count unclosed brackets and try to close them
    repaired = blob
    open_curly = repaired.count("{") - repaired.count("}")
    open_bracket = repaired.count("[") - repaired.count("]")
    if open_bracket > 0:
        repaired += "]" * open_bracket
    if open_curly > 0:
        repaired += "}" * open_curly

    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            raw_cmds = obj.get("commands", [])
            if isinstance(raw_cmds, list) and raw_cmds:
                commands = _flatten_commands(raw_cmds)
                if commands:
                    return True, commands, bool(obj.get("analysis")), bool(obj.get("plan")), "repaired_json"
    except Exception:
        pass

    # Strategy 2: Try progressively truncating to find valid JSON
    # (sometimes the last few chars are garbled)
    for cut in range(len(blob) - 1, max(len(blob) - 100, start), -1):
        if blob[cut] in ",[":
            candidate = blob[:cut]
            oc = candidate.count("{") - candidate.count("}")
            ob = candidate.count("[") - candidate.count("]")
            candidate += "]" * max(ob, 0) + "}" * max(oc, 0)
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    raw_cmds = obj.get("commands", [])
                    if isinstance(raw_cmds, list) and raw_cmds:
                        commands = _flatten_commands(raw_cmds)
                        if commands:
                            return True, commands, bool(obj.get("analysis")), bool(obj.get("plan")), "partial_repair"
            except Exception:
                continue

    # Strategy 3: Keystroke regex (same as fallback_commands, but on preview)
    keystroke_cmds = _extract_keystrokes(text)
    if len(keystroke_cmds) > len(existing_units):
        return False, keystroke_cmds, has_analysis, has_plan, "more_keystrokes"

    return False, existing_units, has_analysis, has_plan, "kept_original"


def _flatten_commands(raw_cmds: list) -> list[str]:
    commands = []
    for cmd in raw_cmds:
        if isinstance(cmd, dict):
            ks = str(cmd.get("keystrokes", ""))
            pieces = [p.strip() for p in ks.replace("\r\n", "\n").replace("\r", "\n").split("\n") if p.strip()]
            if pieces:
                commands.extend(pieces)
            else:
                commands.append("<WAIT>")
        elif isinstance(cmd, str) and cmd.strip():
            commands.append(cmd.strip())
    return commands


def _extract_keystrokes(text: str) -> list[str]:
    commands = []
    for m in re.finditer(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        try:
            decoded = bytes(m.group(1), "utf-8").decode("unicode_escape")
        except Exception:
            decoded = m.group(1)
        pieces = [p.strip() for p in decoded.replace("\r\n", "\n").replace("\r", "\n").split("\n") if p.strip()]
        commands.extend(pieces)
    # Truncated keystroke string
    for m in re.finditer(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)$', text):
        try:
            decoded = bytes(m.group(1), "utf-8").decode("unicode_escape")
        except Exception:
            decoded = m.group(1)
        pieces = [p.strip() for p in decoded.replace("\r\n", "\n").replace("\r", "\n").split("\n") if p.strip()]
        commands.extend(pieces)
    return commands


# ── Main ─────────────────────────────────────────────────────────────────────

def process_file(input_jsonl: Path, output_json: Path) -> None:
    steps = []
    with open(input_jsonl) as f:
        for line in f:
            steps.append(json.loads(line))

    n = len(steps)
    print(f"=== DeepSeek-V4-Pro Post-Processing ===")
    print(f"Input: {input_jsonl.name} ({n} steps)\n")

    original_score = sum(s["command_f1"] for s in steps) / n * 100
    original_valid = sum(1 for s in steps if s["valid_json"]) / n * 100
    original_exact = sum(1 for s in steps if s["first_cmd_exact"]) / n * 100
    print(f"Original:  Score={original_score:.2f}  Valid={original_valid:.1f}%  Exact={original_exact:.1f}%")

    enhanced_steps = []
    notes = Counter()
    improved = 0
    worsened = 0

    for s in steps:
        if s["valid_json"]:
            enhanced_steps.append(dict(s))
            notes["already_valid"] += 1
            continue

        valid, new_units, has_a, has_p, note = repair_and_extract(
            s["pred_preview"], s["pred_command_units"]
        )
        new_units = normalize_units(new_units)
        notes[note] += 1

        first_exact, precision, recall, f1 = score_commands(new_units, s["ref_command_units"])

        new_step = dict(s)
        new_step["valid_json"] = valid
        new_step["pred_command_units"] = new_units
        new_step["has_analysis"] = has_a
        new_step["has_plan"] = has_p
        new_step["first_cmd_exact"] = first_exact
        new_step["command_precision"] = round(precision, 4)
        new_step["command_recall"] = round(recall, 4)
        new_step["command_f1"] = round(f1, 4)
        new_step["_pp_note"] = note

        if f1 > s["command_f1"] + 0.001:
            improved += 1
        elif f1 < s["command_f1"] - 0.001:
            worsened += 1

        enhanced_steps.append(new_step)

    new_score = sum(s["command_f1"] for s in enhanced_steps) / n * 100
    new_valid = sum(1 for s in enhanced_steps if s["valid_json"]) / n * 100
    new_exact = sum(1 for s in enhanced_steps if s["first_cmd_exact"]) / n * 100

    print(f"Post-PP:   Score={new_score:.2f}  Valid={new_valid:.1f}%  Exact={new_exact:.1f}%")
    print(f"Delta:     {new_score - original_score:+.2f}  (improved={improved}, worsened={worsened})")

    print(f"\nRepair notes:")
    for note, count in notes.most_common():
        print(f"  {note}: {count}")

    # Theoretical analysis: what if all JSON were valid?
    valid_steps = [s for s in steps if s["valid_json"]]
    if valid_steps:
        valid_f1 = sum(s["command_f1"] for s in valid_steps) / len(valid_steps)
        theoretical = valid_f1 * 100
        print(f"\n=== Theoretical Analysis ===")
        print(f"Valid JSON steps ({len(valid_steps)}) avg F1: {valid_f1:.4f}")
        print(f"If ALL steps had that F1: Score = {theoretical:.2f}")
        print(f"Current Score: {original_score:.2f}")
        print(f"Gap: {theoretical - original_score:+.2f}")

    # Bucket comparison
    print(f"\n=== Bucket ===")
    for b in ["early", "mid", "late"]:
        orig = [s for s in steps if s["bucket"] == b]
        enh = [s for s in enhanced_steps if s["bucket"] == b]
        if orig:
            of1 = sum(s["command_f1"] for s in orig) / len(orig)
            nf1 = sum(s["command_f1"] for s in enh) / len(enh)
            ov = sum(1 for s in orig if s["valid_json"]) / len(orig) * 100
            nv = sum(1 for s in enh if s["valid_json"]) / len(enh) * 100
            print(f"  {b:5s} ({len(orig):3d}): F1 {of1:.4f}->{nf1:.4f} ({nf1-of1:+.4f}) | Valid {ov:.1f}%->{nv:.1f}%")

    # Save
    result = {
        "postprocessed": True,
        "source_file": str(input_jsonl),
        "timestamp": datetime.now().isoformat(),
        "original_score": round(original_score, 2),
        "postprocessed_score": round(new_score, 2),
        "delta": round(new_score - original_score, 2),
        "improved_steps": improved,
        "worsened_steps": worsened,
        "repair_notes": dict(notes),
        "steps": n,
        "aggregate": {
            "steps": n,
            "valid_json_pct": round(new_valid, 1),
            "first_cmd_exact_pct": round(new_exact, 1),
            "avg_command_f1": round(new_score / 100, 4),
            "avg_command_precision": round(sum(s["command_precision"] for s in enhanced_steps) / n, 4),
            "avg_command_recall": round(sum(s["command_recall"] for s in enhanced_steps) / n, 4),
            "next_action_score": round(new_score, 2),
        },
        "per_step": enhanced_steps,
    }
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_json}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <incr.jsonl> [output.json]")
        sys.exit(1)
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path.parent / f"{input_path.stem.replace('.incr', '')}_postprocessed.json"
    process_file(input_path, output_path)
