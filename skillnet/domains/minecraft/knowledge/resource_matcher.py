"""
Minecraft Resource Matcher

Minecraft resource matcher - loads resources from minecraft-data and matches against skill names.

migrated from optimizer_impl.py
"""

import os
import re
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Set, Optional

from .minecraft_items import COMMON_BLOCKS, COMMON_ITEMS, RESOURCE_ALIASES

# Minecraft-specific defaults (previously on optimizer BloatProtectionConfig)
_DEFAULT_MC_VERSION = "1.19"
_DEFAULT_CACHE_DIR = ".cache"
_DEFAULT_CACHE_ENABLED = True


class MinecraftResourceMatcher:
    """Loads resources from minecraft-data and matches against skill names.

    Used to detect whether the skill code references resources that don't match the skill name.
    For example: mineCobblestone should not include coal_ore as a mining target.
    """

    # Class-level cache (shared across all instances)
    _cached_blocks: Optional[Set[str]] = None
    _cached_items: Optional[Set[str]] = None
    _cached_version: Optional[str] = None

    def __init__(self, mc_version: str = None, cache_dir: str = None):
        self.mc_version = mc_version or _DEFAULT_MC_VERSION
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", _DEFAULT_CACHE_DIR
        )
        self._cache_file = os.path.join(self.cache_dir, f"mcdata_{self.mc_version}.json")
        self._blocks: Optional[Set[str]] = None
        self._items: Optional[Set[str]] = None
        self._resource_groups: Optional[Dict[str, List[str]]] = None
    
    def _load_mcdata(self):
        """Load minecraft-data, with caching"""
        # 1. Check class-level cache (in-memory)
        if (MinecraftResourceMatcher._cached_blocks is not None and
            MinecraftResourceMatcher._cached_version == self.mc_version):
            self._blocks = MinecraftResourceMatcher._cached_blocks
            self._items = MinecraftResourceMatcher._cached_items
            self._build_resource_groups()
            return

        # 2. Check file cache
        if _DEFAULT_CACHE_ENABLED and os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, 'r') as f:
                    data = json.load(f)
                    if data.get("version") == self.mc_version:
                        self._blocks = set(data["blocks"])
                        self._items = set(data["items"])
                        # Update class-level cache
                        MinecraftResourceMatcher._cached_blocks = self._blocks
                        MinecraftResourceMatcher._cached_items = self._items
                        MinecraftResourceMatcher._cached_version = self.mc_version
                        print(f"[MCData] Loaded {len(self._blocks)} blocks, {len(self._items)} items from cache")
                        self._build_resource_groups()
                        return
            except Exception as e:
                print(f"[MCData] Failed to load cache: {e}")

        # 3. Fetch from Node.js and cache
        try:
            script = f'''
const mcData = require("minecraft-data")("{self.mc_version}");
console.log(JSON.stringify({{
    blocks: Object.keys(mcData.blocksByName),
    items: Object.keys(mcData.itemsByName)
}}));
'''
            # Run inside the mineflayer directory, since minecraft-data is installed there
            mineflayer_dir = os.path.join(os.path.dirname(__file__), "..", "action_space", "env", "mineflayer")
            mineflayer_dir = os.path.abspath(mineflayer_dir)
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True, text=True, timeout=10,
                cwd=mineflayer_dir if os.path.exists(mineflayer_dir) else None
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                self._blocks = set(data["blocks"])
                self._items = set(data["items"])

                # Save to file cache
                if _DEFAULT_CACHE_ENABLED:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    with open(self._cache_file, 'w') as f:
                        json.dump({
                            "version": self.mc_version,
                            "blocks": list(self._blocks),
                            "items": list(self._items),
                            "cached_at": datetime.now().isoformat()
                        }, f)
                    print(f"[MCData] Cached to {self._cache_file}")

                # Update class-level cache
                MinecraftResourceMatcher._cached_blocks = self._blocks
                MinecraftResourceMatcher._cached_items = self._items
                MinecraftResourceMatcher._cached_version = self.mc_version
                print(f"[MCData] Loaded {len(self._blocks)} blocks, {len(self._items)} items from Node.js")
            else:
                raise Exception(f"Node.js execution failed: {result.stderr}")

        except Exception as e:
            print(f"[MCData] Node.js load failed: {e}; using fallback data")
            self._blocks = self._get_fallback_blocks()
            self._items = self._get_fallback_items()

        self._build_resource_groups()

    def _get_fallback_blocks(self) -> Set[str]:
        """Use the block list from shared constants (fallback).

        When Node.js/minecraft-data is unavailable, COMMON_BLOCKS is used as a fallback.
        This gives broader coverage than the previously hardcoded 25 blocks.
        """
        # Use COMMON_BLOCKS from the shared constants
        return COMMON_BLOCKS.copy()

    def _get_fallback_items(self) -> Set[str]:
        """Use the item list from shared constants (fallback).

        When Node.js/minecraft-data is unavailable, COMMON_ITEMS is used as a fallback.
        This gives broader coverage than the previously hardcoded 16 items.
        """
        # Use COMMON_ITEMS from the shared constants
        return COMMON_ITEMS.copy()

    def _build_resource_groups(self):
        """Build resource groups (e.g., logs contains every type of log)"""
        if self._blocks is None:
            return
            
        self._resource_groups = {
            "logs": [b for b in self._blocks if b.endswith("_log")],
            "planks": [b for b in self._blocks if b.endswith("_planks")],
            "ores": [b for b in self._blocks if b.endswith("_ore")],
            "stones": ["stone", "cobblestone", "deepslate", "cobbled_deepslate", "blackstone"],
            "wood": [b for b in self._blocks if "_log" in b or "_planks" in b],
        }
    
    def extract_resource_from_skill_name(self, skill_name: str) -> List[str]:
        """Extract the expected resource from a skill name.

        Args:
            skill_name: skill name, e.g., "mineCobblestone", "craftFurnace"

        Returns:
            List of expected resources, e.g., ["cobblestone", "stone"]
        """
        self._load_mcdata()

        skill_lower = skill_name.lower()

        # Strip common prefixes
        resource_part = skill_lower
        for prefix in ["mine", "craft", "smelt", "collect", "get", "obtain"]:
            if skill_lower.startswith(prefix):
                resource_part = skill_lower[len(prefix):]
                break

        # Strip quantity prefixes (e.g., "threeiron" -> "iron")
        number_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        for num in number_words:
            if resource_part.startswith(num):
                resource_part = resource_part[len(num):]
                break

        # Use the resource-alias mapping from the shared constants
        # Dynamically add logs (use the dynamic list if mcdata is available)
        dynamic_logs = [b for b in (self._blocks or set()) if b.endswith("_log")]
        if dynamic_logs:
            # Prefer the dynamic logs from mcdata
            local_aliases = {"logs": dynamic_logs, "log": dynamic_logs}
        else:
            local_aliases = {}

        # Check the shared-constant resource alias map
        if resource_part in RESOURCE_ALIASES:
            return RESOURCE_ALIASES[resource_part]
        # Check the dynamic map (overrides the static values in the shared constants)
        if resource_part in local_aliases:
            return local_aliases[resource_part]

        # Try a direct match
        if self._blocks and resource_part in self._blocks:
            return [resource_part]
        if self._items and resource_part in self._items:
            return [resource_part]

        # Try matching a resource group
        if self._resource_groups:
            for group_name, resources in self._resource_groups.items():
                if group_name in resource_part:
                    return resources

        # Try fuzzy matching (e.g., "ironore" -> "iron_ore")
        if self._blocks:
            normalized = resource_part.replace("_", "")
            for block in self._blocks:
                if block.replace("_", "") == normalized:
                    return [block]

        return []  # unknown resource; skip the check

    def is_resource_in_code(self, code: str, expected_resources: List[str]) -> bool:
        """Check whether the code uses the expected resources"""
        code_lower = code.lower()
        return any(res in code_lower for res in expected_resources)

    def find_unexpected_resources(self, code: str, expected_resources: List[str]) -> List[str]:
        """Find unexpected resources that appear in the code.

        Inspect the blocks parameter of mineBlocks calls and identify resources that don't match expectations.

        Args:
            code: skill code
            expected_resources: list of expected resources

        Returns:
            List of unexpected resources
        """
        self._load_mcdata()
        unexpected = []

        if not self._blocks:
            return unexpected

        # Inspect the blocks parameter in mineBlocks calls
        for m in re.finditer(r"blocks\s*:\s*\[([^\]]+)\]", code):
            blocks_str = m.group(1)
            for block in self._blocks:
                if block in blocks_str and block not in expected_resources:
                    # Check whether it is a completely unrelated resource
                    if not self._is_related_resource(block, expected_resources):
                        if block not in unexpected:
                            unexpected.append(block)

        return unexpected

    def _is_related_resource(self, resource: str, expected: List[str]) -> bool:
        """Check whether a resource is related to any of the expected resources.

        For example: iron_ore and raw_iron are related; coal_ore and coal are related.
        """
        # Extract the base names
        base_names = set()
        for exp in expected:
            base = exp.replace("_ore", "").replace("raw_", "").replace("_ingot", "")
            base = base.replace("deepslate_", "")
            base_names.add(base)
        
        resource_base = resource.replace("_ore", "").replace("raw_", "").replace("_ingot", "")
        resource_base = resource_base.replace("deepslate_", "")
        
        return resource_base in base_names
