"""
Canonical deposit classification for the full-inventory "deposit useless items"
flow. Single source of truth, replacing the per-skill hardcoded block whitelist.

"Useless" = low value for unlocking new items / advancing progression. An item is
deposited iff it falls into one of:
  - OBSOLETE tool/armor: a lower-tier item of a type for which a higher tier is
    held in inventory OR worn (you keep the best pickaxe, deposit the rest).
  - JUNK: decorative/byproduct blocks that are rarely recipe ingredients.
  - EXCESS: a common building block held beyond a conservative reserve.
Everything else is KEPT, including any UNKNOWN item (conservative: never deposit
something that might matter). Depositing is to a chest (non-destructive), so being
moderately aggressive is safe.

Static, rule-based (item-name patterns + small sets); no recipe-graph dependency.
"""
import math
import re

STACK_SIZE = 64


def _slots(count):
    return math.ceil(count / STACK_SIZE) if count > 0 else 0


# ----- tool / armor tier model -----
# Tool types whose lower tiers become obsolete once a higher tier is held/worn.
_TOOL_TYPES = ("pickaxe", "axe", "sword", "shovel", "hoe")
_ARMOR_TYPES = ("helmet", "chestplate", "leggings", "boots")
# Tier rank (higher = better / supersedes lower). Covers both tool and armor
# material prefixes; unknown prefixes rank 0.
_TIER = {
    "wooden": 1, "wood": 1, "leather": 1,
    "golden": 2, "gold": 2, "chainmail": 2, "chain": 2,
    "stone": 3,
    "iron": 4,
    "diamond": 5,
    "netherite": 6,
}


def _typed_item(name):
    """Return (type, tier) if `name` is a tool or armor piece, else (None, 0)."""
    for t in _TOOL_TYPES + _ARMOR_TYPES:
        if name == t or name.endswith("_" + t):
            prefix = name[: -(len(t) + 1)] if name.endswith("_" + t) else ""
            return t, _TIER.get(prefix, 0)
    return None, 0


# ----- JUNK: decorative / byproduct, rarely a recipe ingredient -----
_JUNK = {
    "granite", "andesite", "diorite", "tuff", "dirt", "coarse_dirt", "gravel",
    "cobbled_deepslate", "cobbled_deepstone", "deepslate", "netherrack",
    "flint", "seeds", "wheat_seeds", "rotten_flesh", "poisonous_potato",
    "oak_sapling", "birch_sapling", "spruce_sapling", "jungle_sapling",
    "acacia_sapling", "dark_oak_sapling", "mangrove_propagule", "cherry_sapling",
}


# ----- EXCESS: useful in moderation; deposit beyond a conservative reserve -----
_EXCESS_RESERVE = {
    "cobblestone": 64,
    "stick": 16,
}
# all plank/log variants share a reserve
_PLANK_RESERVE = 32
_LOG_RESERVE = 16
# crafting stations: keep a small working stock
_STATION_RESERVE = {"crafting_table": 2, "furnace": 2, "blast_furnace": 2, "smoker": 2}


def _excess_reserve(name):
    if name in _EXCESS_RESERVE:
        return _EXCESS_RESERVE[name]
    if name in _STATION_RESERVE:
        return _STATION_RESERVE[name]
    if name.endswith("_planks"):
        return _PLANK_RESERVE
    if name.endswith("_log") or name.endswith("_wood"):
        return _LOG_RESERVE
    return None


def classify_deposit(inventory, equipment=None):
    """Classify an inventory into deposit vs keep.

    Args:
        inventory: dict {item_name: count}.
        equipment: optional list of worn-equipment item names (armor/hand); a worn
            piece counts as a held higher tier for obsolescence.

    Returns:
        {"deposit": {item: count_to_deposit}, "freed_slots": int}
        Only items with a positive deposit count are included. freed_slots is the
        number of inventory slots a perfect deposit of that set would free.
    """
    inventory = {k: int(v) for k, v in (inventory or {}).items() if int(v) > 0}
    worn = [e for e in (equipment or []) if e]

    # Best tier held (inventory + worn) per tool/armor type.
    best_tier = {}
    for name in list(inventory.keys()) + worn:
        t, tier = _typed_item(name)
        if t is not None:
            best_tier[t] = max(best_tier.get(t, 0), tier)

    deposit = {}
    for name, count in inventory.items():
        t, tier = _typed_item(name)
        # 1) OBSOLETE tool/armor: a strictly-higher tier of this type is held/worn.
        if t is not None:
            if tier < best_tier.get(t, 0):
                deposit[name] = count
            continue  # tools/armor never fall through to junk/excess
        # 2) JUNK
        if name in _JUNK:
            deposit[name] = count
            continue
        # 3) EXCESS beyond reserve
        reserve = _excess_reserve(name)
        if reserve is not None and count > reserve:
            deposit[name] = count - reserve
            continue
        # 4) otherwise KEEP (valuables, food, utility, best tools, unknown)

    freed = 0
    for name, dc in deposit.items():
        freed += _slots(inventory[name]) - _slots(inventory[name] - dc)
    return {"deposit": deposit, "freed_slots": freed}
