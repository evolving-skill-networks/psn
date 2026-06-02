"""
Factual primitive-doc injection for Phase 1 analyzer prompts.

Phase 13.B-9 R3/R4 finding: Phase 1 LLM analyzers (Qwen3-Coder-Next-FP8 and
gpt-5-mini) hallucinate proximity-validation gradients (e.g., "add
distanceTo > 4 → skip" before mineBlock calls) at ~60% rate even when
the chat log has no proximity error. The hallucination originates in
the LLM's pretrained association `mining + distance > N → proximity bug`
and is robust to Tier-1 evidence-grounded prompt constraints.

This module provides Tier-2 factual API behavior docs that are injected
into the Phase 1 analysis prompt only for primitives whose call sites
appear in the skill being analyzed. The docs state what each primitive
actually does (e.g., "mineBlock... navigates the bot via
mineflayer-pathfinder"); they do NOT contain anti-pattern instructions
("do NOT add proximity check..."), keeping the wording purely factual.

R4 result on n=10 trials × 5 scenarios × 2 LLMs (40 trials per condition
per model on the 4 hallucination scenarios):
  Qwen3:    baseline 60% → Tier-2 52% (-8pp)
  gpt-5-mini: baseline 60% → Tier-2 50% (-10pp); chunk 8/10 → 2/10 (p=0.023)
TPR (true_proximity_bug) preserved at 10/10 across all conditions.

Gated by OptimizationConfig.use_factual_primitive_doc (default False).
"""

from __future__ import annotations

import re
from typing import List


_PRIMITIVE_FACTUAL_DOCS = {
    "mineBlock": (
        "mineBlock(bot, name, count) — searches up to ~32 blocks of the bot "
        "for blocks matching `name`, navigates the bot to them via "
        "mineflayer-pathfinder, equips an appropriate tool from inventory, "
        "digs the block, and waits for drops to be picked up. Returns the "
        "count of blocks actually mined. The primitive handles navigation, "
        "tool selection, and drop pickup automatically as part of the call."
    ),
    "exploreUntil": (
        "exploreUntil(bot, direction, maxTime, condition) — moves the bot in "
        "the given direction for up to `maxTime` seconds, periodically "
        "evaluating `condition()`. Returns the truthy value of `condition()` "
        "when it is met, or null/undefined on timeout. The primitive handles "
        "movement and condition polling internally."
    ),
    "placeItem": (
        "placeItem(bot, name, position) — equips the named item from "
        "inventory, navigates the bot to within reach of `position`, and "
        "places the block at that Vec3 against an adjacent support face. "
        "The primitive handles equipping, navigation, and the place action."
    ),
    "craftItem": (
        "craftItem(bot, name, count) — finds a recipe for `name`, locates a "
        "crafting table in range (using one if already nearby) or uses 2x2 "
        "inventory crafting when sufficient, performs the craft, and waits "
        "for the resulting items. Returns when the requested count has been "
        "crafted or the recipe cannot be satisfied."
    ),
    "bot.dig": (
        "bot.dig(block) — raw mineflayer block-dig API. Requires the bot to "
        "already be within interaction range (~4 blocks) of `block`; the API "
        "does not move the bot. Returns a Promise that resolves on dig "
        "completion or rejects if the block is unreachable. Tool equipping "
        "is the caller's responsibility before calling bot.dig."
    ),
}


def detect_primitives_in_code(skill_code: str) -> List[str]:
    """Return a sorted list of primitive names whose call sites appear in
    `skill_code`.

    Detection uses `<name>(` matching with a lookbehind preventing word
    extension, so `mineBlock(` matches but `myMineBlock(` does not. The
    `bot.dig` entry is matched literally with the dot escaped.
    """
    if not skill_code:
        return []
    found: List[str] = []
    for name in _PRIMITIVE_FACTUAL_DOCS:
        pattern = re.compile(rf'(?<!\w){re.escape(name)}\s*\(')
        if pattern.search(skill_code):
            found.append(name)
    return sorted(found)


def build_factual_primitive_section(skill_code: str) -> str:
    """Compose the factual doc block for primitives present in `skill_code`.

    Returns an empty string when no covered primitive is detected (so the
    section is omitted entirely rather than left as an empty header).
    """
    used = detect_primitives_in_code(skill_code)
    if not used:
        return ""
    lines = [
        "",
        "**Primitive Behavior Reference (only primitives this skill calls):**",
    ]
    for name in used:
        lines.append(f"- {_PRIMITIVE_FACTUAL_DOCS[name]}")
    return "\n".join(lines) + "\n"
