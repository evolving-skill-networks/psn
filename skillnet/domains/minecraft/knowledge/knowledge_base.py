"""
MinecraftKnowledgeBase - Symbolic knowledge base for Minecraft recipes and dependencies

Loads and parses minecraft-data to provide:
- Recipe queries (item -> ingredients)
- Dependency tree generation (diamond_pickaxe -> all required raw materials)
- Tool requirements (what tool is needed to mine a block)
- Smelting recipes
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path


@dataclass
class Recipe:
    """A crafting recipe"""
    result: str  # Result item name
    result_count: int  # How many items produced
    ingredients: Dict[str, int]  # item_name -> count required
    requires_crafting_table: bool  # Whether 3x3 crafting grid is needed
    is_shapeless: bool  # Whether the recipe is shapeless


# Ingredient accessibility preference for recipe variant selection.
# When multiple recipe variants have equal missing count, prefer more accessible ingredients.
# Lower score = more preferred (surface-accessible > deep underground > Nether).
_INGREDIENT_PREFERENCE = {
    "cobblestone": 0,
    "cobbled_deepslate": 10,
    "blackstone": 20,
}


def _variant_preference_score(ingredients: Dict[str, int]) -> int:
    """Lower score = more preferred variant. Unknown ingredients get score 5."""
    return sum(_INGREDIENT_PREFERENCE.get(item, 5) for item in ingredients)


@dataclass
class SmeltRecipe:
    """A smelting/furnace recipe"""
    input_item: str
    output_item: str
    output_count: int = 1


@dataclass
class BlockInfo:
    """Information about a block"""
    name: str
    hardness: float
    harvest_tools: List[str]  # Tools that can harvest this block
    min_tool_tier: Optional[str]  # Minimum tool tier required (wood, stone, iron, diamond)
    drops: List[str]  # Items dropped when mined


@dataclass
class DependencyNode:
    """A node in the dependency tree"""
    item: str
    count: int
    children: List['DependencyNode'] = field(default_factory=list)
    is_raw_material: bool = False  # Can be mined/gathered directly
    alternative_sources: List[str] = field(default_factory=list)  # Other ways to obtain

    def to_dict(self) -> Dict:
        return {
            "item": self.item,
            "count": self.count,
            "is_raw_material": self.is_raw_material,
            "alternative_sources": self.alternative_sources,
            "children": [c.to_dict() for c in self.children]
        }

    def flatten(self) -> Dict[str, int]:
        """Flatten the tree to get total raw materials needed"""
        if self.is_raw_material:
            return {self.item: self.count}

        result: Dict[str, int] = {}
        for child in self.children:
            child_materials = child.flatten()
            for item, count in child_materials.items():
                result[item] = result.get(item, 0) + count
        return result


# Tool tiers in order of power
TOOL_TIERS = ["wooden", "stone", "iron", "diamond", "netherite"]

# Raw materials that can be gathered directly (not crafted)
RAW_MATERIALS = {
    # Logs
    "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log",
    "dark_oak_log", "mangrove_log", "cherry_log",
    # Ores and minerals
    "cobblestone", "stone", "coal", "raw_iron", "raw_gold", "raw_copper",
    "diamond", "emerald", "lapis_lazuli", "redstone", "quartz",
    # Other gatherable items
    "sand", "gravel", "clay_ball", "dirt", "flint",
    "string", "leather", "feather", "bone", "gunpowder",
    "wheat", "carrot", "potato", "beetroot", "sugar_cane",
    "bamboo", "kelp", "seagrass",
    # From animals
    "beef", "porkchop", "chicken", "mutton", "rabbit",
    "wool", "egg",
}

# Common alternative sources for items
ALTERNATIVE_SOURCES = {
    "stick": ["dead_bush"],  # Dead bushes drop sticks
    "coal": ["charcoal"],  # Charcoal is equivalent for fuel
    "string": ["cobweb"],  # Cobwebs drop string
    "leather": ["rabbit_hide"],  # 4 rabbit hides = 1 leather
}


class MinecraftKnowledgeBase:
    """
    Minecraft knowledge base for recipes, dependencies, and tool requirements.

    Loads data from minecraft-data npm package.
    """

    def __init__(self, mc_data_path: Optional[str] = None, mc_version: str = "1.20"):
        """
        Initialize the knowledge base.

        Args:
            mc_data_path: Path to minecraft-data. If None, uses default location.
            mc_version: Minecraft version to use (default: 1.20)
        """
        if mc_data_path is None:
            # Default path relative to PSN installation
            mc_data_path = os.path.join(
                os.path.dirname(__file__),
                "..", "action_space", "env", "mineflayer", "node_modules",
                "minecraft-data", "minecraft-data", "data", "pc", mc_version
            )

        self.mc_data_path = Path(mc_data_path)
        self.mc_version = mc_version

        # Data stores
        self.items: Dict[int, str] = {}  # id -> name
        self.items_by_name: Dict[str, int] = {}  # name -> id
        self.recipes: Dict[str, List[Recipe]] = {}  # item_name -> recipes
        self.smelt_recipes: Dict[str, SmeltRecipe] = {}  # input_item -> recipe
        self.blocks: Dict[str, BlockInfo] = {}  # block_name -> info

        # Load data
        self._load_items()
        self._load_recipes()
        self._load_blocks()
        self._load_smelt_recipes()

    def _load_items(self) -> None:
        """Load items.json to build id <-> name mappings"""
        items_path = self.mc_data_path / "items.json"
        if not items_path.exists():
            print(f"Warning: items.json not found at {items_path}")
            return

        with open(items_path, 'r') as f:
            items_data = json.load(f)

        for item in items_data:
            item_id = item["id"]
            item_name = item["name"]
            self.items[item_id] = item_name
            self.items_by_name[item_name] = item_id

    def _load_recipes(self) -> None:
        """Load recipes.json and convert to name-based recipes"""
        recipes_path = self.mc_data_path / "recipes.json"
        if not recipes_path.exists():
            print(f"Warning: recipes.json not found at {recipes_path}")
            return

        with open(recipes_path, 'r') as f:
            recipes_data = json.load(f)

        for result_id_str, recipe_list in recipes_data.items():
            result_id = int(result_id_str)
            result_name = self.items.get(result_id)
            if result_name is None:
                continue

            if result_name not in self.recipes:
                self.recipes[result_name] = []

            for recipe_data in recipe_list:
                recipe = self._parse_recipe(recipe_data, result_name)
                if recipe:
                    self.recipes[result_name].append(recipe)

    def _parse_recipe(self, recipe_data: Dict, result_name: str) -> Optional[Recipe]:
        """Parse a single recipe from minecraft-data format"""
        result_count = recipe_data.get("result", {}).get("count", 1)
        if isinstance(recipe_data.get("result"), dict):
            result_count = recipe_data["result"].get("count", 1)
        else:
            result_count = 1

        ingredients: Dict[str, int] = {}
        is_shapeless = False
        requires_table = False

        if "inShape" in recipe_data:
            # Shaped recipe
            shape = recipe_data["inShape"]
            requires_table = len(shape) > 2 or any(len(row) > 2 for row in shape)

            for row in shape:
                for item_id in row:
                    if item_id is None or item_id == 767:  # 767 is often air/empty
                        continue
                    item_name = self.items.get(item_id)
                    if item_name:
                        ingredients[item_name] = ingredients.get(item_name, 0) + 1

        elif "ingredients" in recipe_data:
            # Shapeless recipe
            is_shapeless = True
            for item_id in recipe_data["ingredients"]:
                if item_id is None or item_id == 767:
                    continue
                item_name = self.items.get(item_id)
                if item_name:
                    ingredients[item_name] = ingredients.get(item_name, 0) + 1
            requires_table = len(recipe_data["ingredients"]) > 4

        if not ingredients:
            return None

        return Recipe(
            result=result_name,
            result_count=result_count,
            ingredients=ingredients,
            requires_crafting_table=requires_table,
            is_shapeless=is_shapeless
        )

    def _load_blocks(self) -> None:
        """Load blocks.json for harvest tool requirements"""
        blocks_path = self.mc_data_path / "blocks.json"
        if not blocks_path.exists():
            print(f"Warning: blocks.json not found at {blocks_path}")
            return

        with open(blocks_path, 'r') as f:
            blocks_data = json.load(f)

        for block in blocks_data:
            name = block["name"]
            hardness = block.get("hardness", 0)

            # Determine harvest tools based on material
            harvest_tools = []
            min_tier = None

            # Simple heuristics based on block name
            if "ore" in name or name in ["obsidian", "ancient_debris"]:
                harvest_tools = ["pickaxe"]
                if "iron" in name or "gold" in name or "lapis" in name:
                    min_tier = "stone"
                elif "diamond" in name or "emerald" in name or "redstone" in name:
                    min_tier = "iron"
                elif "ancient_debris" in name:
                    min_tier = "diamond"
            elif "log" in name or "wood" in name or "planks" in name:
                harvest_tools = ["axe"]
            elif "dirt" in name or "sand" in name or "gravel" in name:
                harvest_tools = ["shovel"]
            elif "stone" in name or "cobblestone" in name:
                harvest_tools = ["pickaxe"]

            self.blocks[name] = BlockInfo(
                name=name,
                hardness=hardness,
                harvest_tools=harvest_tools,
                min_tool_tier=min_tier,
                drops=[name]  # Simplified - actual drops are more complex
            )

    def _load_smelt_recipes(self) -> None:
        """Define common smelting recipes (not in minecraft-data by default)"""
        # Common smelting recipes
        smelt_recipes = [
            ("raw_iron", "iron_ingot"),
            ("raw_gold", "gold_ingot"),
            ("raw_copper", "copper_ingot"),
            ("iron_ore", "iron_ingot"),
            ("gold_ore", "gold_ingot"),
            ("copper_ore", "copper_ingot"),
            ("sand", "glass"),
            ("cobblestone", "stone"),
            ("stone", "smooth_stone"),
            ("clay_ball", "brick"),
            ("netherrack", "nether_brick"),
            ("oak_log", "charcoal"),
            ("birch_log", "charcoal"),
            ("spruce_log", "charcoal"),
            ("jungle_log", "charcoal"),
            ("acacia_log", "charcoal"),
            ("dark_oak_log", "charcoal"),
            ("beef", "cooked_beef"),
            ("porkchop", "cooked_porkchop"),
            ("chicken", "cooked_chicken"),
            ("mutton", "cooked_mutton"),
            ("rabbit", "cooked_rabbit"),
            ("cod", "cooked_cod"),
            ("salmon", "cooked_salmon"),
            ("potato", "baked_potato"),
            ("kelp", "dried_kelp"),
        ]

        for input_item, output_item in smelt_recipes:
            self.smelt_recipes[input_item] = SmeltRecipe(
                input_item=input_item,
                output_item=output_item
            )

    # ==================== Query Methods ====================

    def get_recipe(self, item: str, include_decomposition: bool = False) -> Optional[Recipe]:
        """
        Get the primary recipe for an item, preferring synthesis over decomposition.

        Args:
            item: The item to get recipe for
            include_decomposition: If True, include decomposition recipes (default False)

        Decomposition recipes (e.g., 1 iron_block -> 9 iron_ingot) are excluded by default
        because they create circular dependencies when used for resource acquisition.
        """
        recipes = self.recipes.get(item, [])
        if not recipes:
            return None

        if not include_decomposition:
            # Filter out decomposition recipes
            synthesis_recipes = [r for r in recipes if not self._is_decomposition_recipe(r)]
            if synthesis_recipes:
                return synthesis_recipes[0]

        return recipes[0]

    def _is_decomposition_recipe(self, recipe: Recipe) -> bool:
        """
        Check if a recipe is a decomposition recipe.

        Decomposition recipes have these characteristics:
        1. Single ingredient type (total count = 1)
        2. Multiple outputs (result_count > 1)
        3. The output item can be crafted back into the input (reverse recipe exists)

        Examples:
            - 1 iron_block -> 9 iron_ingot (True: 9 ingot -> 1 block exists)
            - 3 iron_ingot + 2 stick -> 1 iron_pickaxe (False: not decomposition)
        """
        # Condition 1 & 2: Single ingredient + multiple outputs
        total_ingredients = sum(recipe.ingredients.values())
        if not (total_ingredients == 1 and recipe.result_count > 1):
            return False

        # Condition 3: Check if reverse recipe exists
        input_item = list(recipe.ingredients.keys())[0]
        reverse_recipes = self.recipes.get(input_item, [])
        for rev in reverse_recipes:
            if recipe.result in rev.ingredients:
                return True  # Reverse recipe exists, confirmed decomposition

        return False

    def get_all_recipes(self, item: str) -> List[Recipe]:
        """Get all recipes that produce an item"""
        return self.recipes.get(item, [])

    def get_smelt_recipe(self, input_item: str) -> Optional[SmeltRecipe]:
        """Get smelting recipe for an input item"""
        return self.smelt_recipes.get(input_item)

    def get_smelt_recipe_for_output(self, output_item: str) -> Optional[SmeltRecipe]:
        """Find what can be smelted to get an output item"""
        for recipe in self.smelt_recipes.values():
            if recipe.output_item == output_item:
                return recipe
        return None

    def is_raw_material(self, item: str) -> bool:
        """Check if an item is a raw material that can be gathered directly"""
        return item in RAW_MATERIALS

    def get_required_tool(self, block: str) -> Optional[str]:
        """Get the tool required to harvest a block"""
        block_info = self.blocks.get(block)
        if block_info and block_info.harvest_tools:
            return block_info.harvest_tools[0]
        return None

    def get_min_tool_tier(self, block: str) -> Optional[str]:
        """Get the minimum tool tier required to harvest a block"""
        block_info = self.blocks.get(block)
        return block_info.min_tool_tier if block_info else None

    # ==================== Dependency Analysis ====================

    def get_dependencies(self, item: str, count: int = 1,
                         visited: Optional[Set[str]] = None) -> DependencyNode:
        """
        Recursively build a dependency tree for an item.

        Args:
            item: The item to analyze
            count: How many of the item are needed
            visited: Set of items already visited (to prevent cycles)

        Returns:
            DependencyNode with full dependency tree
        """
        if visited is None:
            visited = set()

        node = DependencyNode(
            item=item,
            count=count,
            is_raw_material=self.is_raw_material(item),
            alternative_sources=ALTERNATIVE_SOURCES.get(item, [])
        )

        # If raw material, no further dependencies
        if node.is_raw_material:
            return node

        # Check for smelting recipe first
        smelt_recipe = self.get_smelt_recipe_for_output(item)
        if smelt_recipe and smelt_recipe.input_item not in visited:
            # Item is obtained by smelting
            visited.add(item)
            child = self.get_dependencies(
                smelt_recipe.input_item,
                count,
                visited.copy()
            )
            node.children.append(child)
            return node

        # Check for crafting recipe
        recipe = self.get_recipe(item)
        if recipe is None:
            # No recipe found - might be a raw material we didn't recognize
            node.is_raw_material = True
            return node

        # Prevent infinite recursion
        if item in visited:
            return node
        visited.add(item)

        # Calculate how many recipes we need to run
        recipes_needed = (count + recipe.result_count - 1) // recipe.result_count

        # Add children for each ingredient
        for ingredient, ing_count in recipe.ingredients.items():
            total_needed = ing_count * recipes_needed
            child = self.get_dependencies(ingredient, total_needed, visited.copy())
            node.children.append(child)

        return node

    def get_raw_materials_needed(self, item: str, count: int = 1) -> Dict[str, int]:
        """
        Get the total raw materials needed to craft an item.

        Returns:
            Dict of raw_material_name -> count_needed
        """
        dep_tree = self.get_dependencies(item, count)
        return dep_tree.flatten()

    def check_can_craft(self, item: str, inventory: Dict[str, int],
                        count: int = 1) -> Tuple[bool, Dict[str, int]]:
        """
        Check if an item can be crafted with current inventory.
        Tries ALL recipe variants and returns the one with fewest missing items.

        Args:
            item: Item to craft
            inventory: Current inventory {item_name: count}
            count: How many to craft

        Returns:
            (can_craft, missing_items)
        """
        all_recipes = self.get_all_recipes(item)
        recipes = [r for r in all_recipes if not self._is_decomposition_recipe(r)]
        if not recipes:
            recipe = self.get_recipe(item)
            if recipe is None:
                return False, {item: count}
            recipes = [recipe]

        best_missing = None
        for recipe in recipes:
            recipes_needed = (count + recipe.result_count - 1) // recipe.result_count
            missing: Dict[str, int] = {}

            for ingredient, ing_count in recipe.ingredients.items():
                total_needed = ing_count * recipes_needed
                have = inventory.get(ingredient, 0)
                if have < total_needed:
                    missing[ingredient] = total_needed - have

            if not missing:
                return True, {}

            if best_missing is None or sum(missing.values()) < sum(best_missing.values()):
                best_missing = missing
            elif sum(missing.values()) == sum(best_missing.values()):
                # Tiebreaker 1: prefer variant whose ingredients match available log types
                # e.g., birch_planks → birch_log, if birch_log in inventory → prefer this variant
                planks_match = False
                for ing in missing:
                    log_type = ing.replace("_planks", "_log")
                    if inventory.get(log_type, 0) > 0:
                        best_missing = missing
                        planks_match = True
                        break
                # Tiebreaker 2: prefer common overworld ingredients
                # e.g., cobblestone > cobbled_deepslate > blackstone
                if not planks_match:
                    if _variant_preference_score(missing) < _variant_preference_score(best_missing):
                        best_missing = missing

        return False, best_missing or {}

    def get_best_recipe(self, item: str, inventory: Dict[str, int],
                        count: int = 1) -> Optional['Recipe']:
        """
        Get the recipe variant that best matches the current inventory.
        Prefers variants whose ingredients are available in inventory.

        Args:
            item: Item to craft
            inventory: Current inventory {item_name: count}
            count: How many to craft

        Returns:
            Best matching Recipe, or None if no recipe exists
        """
        all_recipes = self.get_all_recipes(item)
        recipes = [r for r in all_recipes if not self._is_decomposition_recipe(r)]
        if not recipes:
            return self.get_recipe(item)

        best_recipe = recipes[0]
        best_missing_total = 999999
        for recipe in recipes:
            recipes_needed = (count + recipe.result_count - 1) // recipe.result_count
            total_missing = 0
            for ingredient, ing_count in recipe.ingredients.items():
                total_needed = ing_count * recipes_needed
                have = inventory.get(ingredient, 0)
                if have < total_needed:
                    total_missing += total_needed - have
            if total_missing < best_missing_total:
                best_missing_total = total_missing
                best_recipe = recipe
            elif total_missing == best_missing_total:
                # Tiebreaker 1: prefer variant matching available log types
                planks_match = False
                for ing, _ in recipe.ingredients.items():
                    if "_planks" in ing:
                        log_type = ing.replace("_planks", "_log")
                        if inventory.get(log_type, 0) > 0:
                            best_recipe = recipe
                            best_missing_total = total_missing
                            planks_match = True
                            break
                # Tiebreaker 2: prefer common overworld ingredients
                if not planks_match:
                    if _variant_preference_score(recipe.ingredients) < _variant_preference_score(best_recipe.ingredients):
                        best_recipe = recipe
                        best_missing_total = total_missing
        return best_recipe

    def get_crafting_path(self, target: str, inventory: Dict[str, int],
                          count: int = 1) -> List[Tuple[str, int]]:
        """
        Get the ordered list of items to craft to reach the target.

        Returns list of (item_name, count) in order they should be crafted.
        """
        path: List[Tuple[str, int]] = []
        needed = self.get_raw_materials_needed(target, count)

        # Simple topological sort based on dependencies
        def add_to_path(item: str, cnt: int, visited: Set[str]):
            if item in visited:
                return
            if self.is_raw_material(item):
                return

            visited.add(item)
            recipe = self.get_recipe(item)
            if recipe:
                for ing in recipe.ingredients:
                    add_to_path(ing, recipe.ingredients[ing], visited)
                path.append((item, cnt))

        add_to_path(target, count, set())
        return path
