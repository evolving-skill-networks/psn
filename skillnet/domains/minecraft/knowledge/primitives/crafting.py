"""
Primitive Knowledge - Crafting
craftItem primitive knowledge and recipe dependency analysis

Migrated from primitive_knowledge.py
All fix_hint, fix_strategy, check_hint fields removed
RecipeDependencyAnalyzer retained (pure computation logic)
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# craftItem Primitive Knowledge
# ============================================================

CRAFTING_KNOWLEDGE: List[KnowledgeItem] = [
    # Preconditions
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="craftItem_materials",
        fact=(
            "craftItem requires all recipe materials to be in inventory before calling. "
            "Checks ingredient counts against recipe requirements. "
            "Fails immediately if any ingredient is insufficient."
        ),
        keywords=["craftitem", "materials", "ingredients", "inventory", "recipe"],
        conditions=["calling craftItem"],
        implications=[
            "check inventory before crafting",
            "all materials must be present",
            "fails if any ingredient missing",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="craftItem_crafting_table",
        fact=(
            "craftItem needs a crafting table nearby for 3x3 recipes. "
            "Simple 2x2 recipes (planks, sticks) work in inventory. "
            "Crafting table must be placed and bot within 32 blocks."
        ),
        keywords=["craftitem", "crafting", "table", "3x3", "recipe", "nearby"],
        conditions=["crafting complex items"],
        implications=[
            "simple recipes work in inventory",
            "complex recipes need table",
            "must be within range of table",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="craftItem_recipe_exists",
        fact=(
            "craftItem fails if no recipe exists for the target item. "
            "Some items cannot be crafted (ores, mob drops, special items). "
            "Item name must match Minecraft registry exactly."
        ),
        keywords=["craftitem", "recipe", "exists", "valid", "name"],
        conditions=["crafting any item"],
        implications=[
            "not all items are craftable",
            "use correct item names",
            "underscores not spaces",
        ],
    ),

    # Failure patterns
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="craftItem_insufficient_materials",
        fact=(
            "craftItem fails with 'Insufficient materials' or 'Missing X' when inventory lacks ingredients. "
            "Error message indicates which material is missing. "
            "Must gather or craft missing materials first."
        ),
        keywords=["craftitem", "insufficient", "missing", "materials", "not enough"],
        conditions=["crafting without enough materials"],
        implications=[
            "check which material is missing",
            "gather or craft missing items",
            "calculate total materials needed",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="craftItem_no_table",
        fact=(
            "craftItem fails for 3x3 recipes when no placed crafting_table block is within 32 blocks. "
            "Player inventory only has 2x2 crafting grid. "
            "Error message contains 'No crafting table', '3x3 recipe', or 'no available recipes'. "
            "A crafting_table item in inventory does not satisfy this requirement — "
            "it must be a placed Block in the world."
        ),
        keywords=["craftitem", "no", "crafting", "table", "3x3", "need",
                  "no available recipes", "requires a crafting table"],
        conditions=["3x3 recipe without placed table"],
        logical_implications=[
            "inventory crafting_table ≠ placed crafting_table block",
            "3x3 recipes cannot be crafted with 2x2 inventory grid",
            "crafting_table block must exist within 32 blocks range",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="craftItem_invalid_name",
        fact=(
            "craftItem fails with 'No item named' or 'Invalid item' for incorrect item names. "
            "Minecraft uses underscores: 'oak_planks' not 'oak planks'. "
            "Singular vs plural matters: 'plank' vs 'planks'."
        ),
        keywords=["craftitem", "invalid", "name", "item", "no such"],
        conditions=["using wrong item name"],
        implications=[
            "use underscores not spaces",
            "check exact spelling",
            "singular vs plural matters",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="craftItem_no_recipe",
        fact=(
            "craftItem fails with 'Recipe not found' for non-craftable items. "
            "Some items require smelting (iron ingot from raw iron). "
            "Some items only from mobs or trading (ender pearls, blaze rods)."
        ),
        keywords=["craftitem", "recipe", "not found", "cannot craft"],
        conditions=["trying to craft non-craftable item"],
        implications=[
            "item may need smelting",
            "item may be mob drop",
            "item may be trade-only",
        ],
    ),

    # Internal table search behavior
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="craftItem_internal_table_search",
        fact=(
            "craftItem internally uses bot.findBlock to search for placed crafting_table blocks "
            "within 32 blocks. It passes this found Block to bot.checkRecipe and bot.craft. "
            "When bot.checkRecipe is called externally without a craftingTable argument, "
            "it defaults to null, which means only 2x2 grid recipes are returned. "
            "3x3 recipes (furnace, tools, swords, armor) require a non-null craftingTable Block."
        ),
        keywords=["craftitem", "checkrecipe", "crafting", "table", "findblock",
                  "internal", "32", "3x3", "null", "no available recipes"],
        conditions=["using craftItem or bot.checkRecipe"],
        logical_implications=[
            "external checkRecipe without craftingTable gives different results than craftItem's internal check",
            "3x3 recipes appear unavailable when craftingTable is null",
            "craftItem already handles table lookup — separate checkRecipe call may give false negatives",
        ],
    ),

    # Interchangeable variant crafting
    KnowledgeItem(
        domain=KnowledgeDomain.GAME,
        category=KnowledgeCategory.GAME_MECHANIC,
        name="interchangeable_variant_crafting",
        fact=(
            "Many Minecraft items have interchangeable variants that serve the same purpose: "
            "any *_log crafts into its corresponding *_planks (4 per log), "
            "any planks type can craft sticks/tools/slabs. "
            "When a task requires a generic item (e.g., 'planks' without specifying type), "
            "any variant satisfies the requirement equally."
        ),
        keywords=["variant", "planks", "logs", "interchangeable", "crafting",
                  "oak", "birch", "spruce", "generic", "any type"],
        conditions=["crafting generic items", "choosing between variants"],
        implications=[
            "all log types produce equivalent planks",
            "choosing the wrong variant wastes time if its raw material is scarce",
            "inventory raw material counts determine which variant is most efficient",
        ],
    ),

    # Count parameter behavior
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="craftItem_count_is_recipe_executions",
        fact=(
            "craftItem(bot, name, count) executes the recipe count times in a loop. "
            "Each execution produces output_count items (varies by recipe). "
            "stick recipe: 1 execution consumes 2 planks, produces 4 sticks (output_count=4). "
            "oak_planks recipe: 1 execution consumes 1 oak_log, produces 4 planks (output_count=4). "
            "wooden_pickaxe recipe: 1 execution consumes 3 planks + 2 sticks, produces 1 pickaxe (output_count=1). "
            "Total materials consumed = count × recipe ingredients per execution. "
            "Total items produced = count × output_count."
        ),
        keywords=["craftitem", "count", "recipe", "execution", "output_count",
                  "stick", "planks", "materials", "insufficient", "ran out"],
        conditions=[
            "craftItem count parameter",
            "material consumption calculation",
            "Crafted X/Y before running out of materials",
        ],
        logical_implications=[
            "craftItem(bot, 'stick', 1) consumes 2 planks, produces 4 sticks",
            "craftItem(bot, 'stick', 4) consumes 8 planks, produces 16 sticks",
            "craftItem(bot, 'oak_planks', 1) consumes 1 log, produces 4 planks",
            "craftItem(bot, 'oak_planks', 2) consumes 2 logs, produces 8 planks",
        ],
    ),

    # Unconditional execution behavior
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="craftItem_always_executes",
        fact=(
            "craftItem(bot, name, count) always executes the recipe count times "
            "regardless of how many of the target item already exist in inventory. "
            "It does NOT check current inventory for the output item before crafting. "
            "For example, if inventory already has 32 sticks and craftItem(bot, 'stick', 4) is called, "
            "it will still execute 4 times, consuming 8 planks and producing 16 more sticks (total 48). "
            "craftItem only checks if INPUT MATERIALS are sufficient, not if OUTPUT is needed."
        ),
        keywords=["craftitem", "inventory", "existing", "already", "unconditional",
                  "always", "execute", "unnecessary", "waste"],
        conditions=[
            "crafting items that may already exist in inventory",
            "unnecessary material consumption",
        ],
        logical_implications=[
            "craftItem does not skip execution even if output item count is already sufficient",
            "materials are consumed even when the crafted items are not needed",
            "inventory quantity of the output item has no effect on craftItem behavior",
        ],
    ),

    # Material calculation
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="craftItem_material_calculation",
        fact=(
            "Material calculation must account for intermediate crafting steps. "
            "Example: wooden_pickaxe needs 3 planks + 2 sticks, but sticks need 2 planks. "
            "Total planks needed: 3 + 2 = 5 if no sticks in inventory. "
            "Common bug: code checks planks >= 3 first, then crafts sticks (consuming planks), "
            "leaving insufficient planks for the final craft. "
            "Must calculate TOTAL planks needed (including for sticks) BEFORE any crafting."
        ),
        keywords=["craftitem", "materials", "calculation", "intermediate", "total",
                  "insufficient", "planks", "sticks", "not enough", "need", "have"],
        conditions=["planning crafting"],
        implications=[
            "include intermediate materials",
            "sticks consume planks",
            "calculate before gathering",
        ],
    ),

    # Recipe variant system
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="craftItem_recipe_variants",
        fact=(
            "Minecraft items that use generic material categories have multiple recipe variants. "
            "wooden_pickaxe has 9 separate recipes in minecraft-data: one per plank type "
            "(oak_planks, birch_planks, spruce_planks, jungle_planks, acacia_planks, "
            "dark_oak_planks, crimson_planks, warped_planks, mangrove_planks). "
            "Each recipe requires 3 of the SAME plank type + 2 sticks. "
            "bot.recipesFor(itemId, null, craftingTable) returns only recipes whose "
            "ingredients are fully available in inventory. "
            "bot.recipesAll(itemId, null, craftingTable) returns all 9 variants regardless of inventory."
        ),
        keywords=["recipe", "variant", "planks", "oak_planks", "birch_planks",
                  "spruce_planks", "wooden", "recipesFor", "recipesAll",
                  "insufficient", "missing"],
        conditions=["crafting wooden tools with plank variants"],
        logical_implications=[
            "each wooden tool recipe requires all planks to be the SAME type",
            "bot.recipesFor() returns empty if no single plank type has enough quantity",
            "bot.recipesAll() returns all 9 variants even when no planks are available",
            "mixing plank types (e.g. 2 birch + 1 oak) does NOT satisfy any single recipe",
        ],
    ),

    # Capability overview (what craftItem handles vs what caller must handle)
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="craftItem_internal_capabilities",
        fact=(
            "craftItem(bot, name, count) is a high-level crafting primitive that internally handles: "
            "1) searching for a placed crafting_table block within 32 blocks (for 3x3 recipes), "
            "2) navigating to the crafting table, 3) opening the crafting UI, "
            "4) executing the craft with the correct recipe. "
            "For 2x2 recipes (planks, sticks, crafting_table itself), no placed table is needed. "
            "For 3x3 recipes (tools, furnace, chest, armor), a placed crafting_table BLOCK must exist "
            "within 32 blocks — the CALLER is responsible for ensuring this. "
            "Compared to raw bot.craft(recipe, count, craftingTable), craftItem handles table lookup "
            "and navigation automatically."
        ),
        keywords=["craftitem", "craft", "capability", "crafting_table", "table", "3x3",
                  "navigate", "bot.craft", "recipe", "2x2"],
        logical_implications=[
            "craftItem handles table lookup and navigation — but table must already be placed",
            "for 3x3 recipes: caller must ensure a crafting_table block is placed within 32 blocks first",
            "for 2x2 recipes: craftItem works without any placed crafting table",
            "manual bot.craft requires the caller to find the table Block and pass it as argument",
        ],
    ),
]


# ============================================================
# Recipe Knowledge (Data)
# ============================================================

@dataclass
class RecipeInfo:
    """Recipe information"""
    output: str
    output_count: int
    ingredients: Dict[str, int]
    requires_table: bool = False


RECIPE_KNOWLEDGE: Dict[str, RecipeInfo] = {
    # Wood products
    "oak_planks": RecipeInfo(
        output="oak_planks", output_count=4,
        ingredients={"oak_log": 1}, requires_table=False,
    ),
    "birch_planks": RecipeInfo(
        output="birch_planks", output_count=4,
        ingredients={"birch_log": 1}, requires_table=False,
    ),
    "spruce_planks": RecipeInfo(
        output="spruce_planks", output_count=4,
        ingredients={"spruce_log": 1}, requires_table=False,
    ),
    "jungle_planks": RecipeInfo(
        output="jungle_planks", output_count=4,
        ingredients={"jungle_log": 1}, requires_table=False,
    ),
    "acacia_planks": RecipeInfo(
        output="acacia_planks", output_count=4,
        ingredients={"acacia_log": 1}, requires_table=False,
    ),
    "dark_oak_planks": RecipeInfo(
        output="dark_oak_planks", output_count=4,
        ingredients={"dark_oak_log": 1}, requires_table=False,
    ),
    "stick": RecipeInfo(
        output="stick", output_count=4,
        ingredients={"oak_planks": 2}, requires_table=False,
    ),
    "crafting_table": RecipeInfo(
        output="crafting_table", output_count=1,
        ingredients={"oak_planks": 4}, requires_table=False,
    ),

    # Tools - Wooden
    "wooden_pickaxe": RecipeInfo(
        output="wooden_pickaxe", output_count=1,
        ingredients={"oak_planks": 3, "stick": 2}, requires_table=True,
    ),
    "wooden_axe": RecipeInfo(
        output="wooden_axe", output_count=1,
        ingredients={"oak_planks": 3, "stick": 2}, requires_table=True,
    ),
    "wooden_shovel": RecipeInfo(
        output="wooden_shovel", output_count=1,
        ingredients={"oak_planks": 1, "stick": 2}, requires_table=True,
    ),
    "wooden_hoe": RecipeInfo(
        output="wooden_hoe", output_count=1,
        ingredients={"oak_planks": 2, "stick": 2}, requires_table=True,
    ),
    "wooden_sword": RecipeInfo(
        output="wooden_sword", output_count=1,
        ingredients={"oak_planks": 2, "stick": 1}, requires_table=True,
    ),

    # Tools - Stone
    "stone_pickaxe": RecipeInfo(
        output="stone_pickaxe", output_count=1,
        ingredients={"cobblestone": 3, "stick": 2}, requires_table=True,
    ),
    "stone_axe": RecipeInfo(
        output="stone_axe", output_count=1,
        ingredients={"cobblestone": 3, "stick": 2}, requires_table=True,
    ),
    "stone_shovel": RecipeInfo(
        output="stone_shovel", output_count=1,
        ingredients={"cobblestone": 1, "stick": 2}, requires_table=True,
    ),
    "stone_sword": RecipeInfo(
        output="stone_sword", output_count=1,
        ingredients={"cobblestone": 2, "stick": 1}, requires_table=True,
    ),

    # Tools - Iron
    "iron_pickaxe": RecipeInfo(
        output="iron_pickaxe", output_count=1,
        ingredients={"iron_ingot": 3, "stick": 2}, requires_table=True,
    ),
    "iron_axe": RecipeInfo(
        output="iron_axe", output_count=1,
        ingredients={"iron_ingot": 3, "stick": 2}, requires_table=True,
    ),
    "iron_shovel": RecipeInfo(
        output="iron_shovel", output_count=1,
        ingredients={"iron_ingot": 1, "stick": 2}, requires_table=True,
    ),
    "iron_sword": RecipeInfo(
        output="iron_sword", output_count=1,
        ingredients={"iron_ingot": 2, "stick": 1}, requires_table=True,
    ),

    # Tools - Diamond
    "diamond_pickaxe": RecipeInfo(
        output="diamond_pickaxe", output_count=1,
        ingredients={"diamond": 3, "stick": 2}, requires_table=True,
    ),
    "diamond_axe": RecipeInfo(
        output="diamond_axe", output_count=1,
        ingredients={"diamond": 3, "stick": 2}, requires_table=True,
    ),
    "diamond_sword": RecipeInfo(
        output="diamond_sword", output_count=1,
        ingredients={"diamond": 2, "stick": 1}, requires_table=True,
    ),

    # Furnace and smelting
    "furnace": RecipeInfo(
        output="furnace", output_count=1,
        ingredients={"cobblestone": 8}, requires_table=True,
    ),
    "blast_furnace": RecipeInfo(
        output="blast_furnace", output_count=1,
        ingredients={"iron_ingot": 5, "furnace": 1, "smooth_stone": 3}, requires_table=True,
    ),

    # Utility
    "torch": RecipeInfo(
        output="torch", output_count=4,
        ingredients={"coal": 1, "stick": 1}, requires_table=False,
    ),
    "chest": RecipeInfo(
        output="chest", output_count=1,
        ingredients={"oak_planks": 8}, requires_table=True,
    ),
    "bucket": RecipeInfo(
        output="bucket", output_count=1,
        ingredients={"iron_ingot": 3}, requires_table=True,
    ),
    "boat": RecipeInfo(
        output="boat", output_count=1,
        ingredients={"oak_planks": 5}, requires_table=True,
    ),

    # Armor
    "iron_helmet": RecipeInfo(
        output="iron_helmet", output_count=1,
        ingredients={"iron_ingot": 5}, requires_table=True,
    ),
    "iron_chestplate": RecipeInfo(
        output="iron_chestplate", output_count=1,
        ingredients={"iron_ingot": 8}, requires_table=True,
    ),
    "iron_leggings": RecipeInfo(
        output="iron_leggings", output_count=1,
        ingredients={"iron_ingot": 7}, requires_table=True,
    ),
    "iron_boots": RecipeInfo(
        output="iron_boots", output_count=1,
        ingredients={"iron_ingot": 4}, requires_table=True,
    ),

    # Special
    "flint_and_steel": RecipeInfo(
        output="flint_and_steel", output_count=1,
        ingredients={"iron_ingot": 1, "flint": 1}, requires_table=False,
    ),
}


# ============================================================
# Recipe Dependency Analyzer
# ============================================================

@dataclass
class MaterialCalculation:
    """Material computation result"""
    target_item: str
    target_count: int
    total_raw_materials: Dict[str, int]
    intermediate_crafts: List[Dict[str, Any]]
    warnings: List[str]


class RecipeDependencyAnalyzer:
    """
    Recipe dependency chain analyzer.

    Solves: when crafting a wooden_pickaxe, you need:
    - 3 planks (directly)
    - 2 sticks (directly)
    - but sticks themselves require 2 planks to craft!

    So actually need: 3 + 2 = 5 planks (if no sticks on hand)
    """

    def __init__(self, recipes: Optional[Dict[str, RecipeInfo]] = None):
        self.recipes = recipes or RECIPE_KNOWLEDGE

    def get_recipe(self, item: str) -> Optional[RecipeInfo]:
        """Get a recipe."""
        return self.recipes.get(item)

    def calculate_materials(
        self,
        target_item: str,
        target_count: int = 1,
        inventory: Optional[Dict[str, int]] = None,
        max_depth: int = 5,
    ) -> MaterialCalculation:
        """
        Compute all materials needed to craft the target item.

        Args:
            target_item: target item
            target_count: required count
            inventory: current inventory {item: count}
            max_depth: maximum recursion depth

        Returns:
            MaterialCalculation: complete material computation result
        """
        inventory = inventory or {}
        result = MaterialCalculation(
            target_item=target_item,
            target_count=target_count,
            total_raw_materials={},
            intermediate_crafts=[],
            warnings=[],
        )

        self._calculate_recursive(
            item=target_item,
            count=target_count,
            inventory=dict(inventory),
            result=result,
            depth=0,
            max_depth=max_depth,
        )

        return result

    def _calculate_recursive(
        self,
        item: str,
        count: int,
        inventory: Dict[str, int],
        result: MaterialCalculation,
        depth: int,
        max_depth: int,
    ):
        """Recursively compute material requirements."""
        if depth > max_depth:
            result.warnings.append(f"Max depth reached for {item}")
            return

        have = inventory.get(item, 0)
        if have >= count:
            inventory[item] = have - count
            return

        need_to_craft = count - have
        inventory[item] = 0

        recipe = self.get_recipe(item)
        if not recipe:
            result.total_raw_materials[item] = (
                result.total_raw_materials.get(item, 0) + need_to_craft
            )
            return

        craft_times = (need_to_craft + recipe.output_count - 1) // recipe.output_count

        result.intermediate_crafts.append({
            "item": item,
            "craft_times": craft_times,
            "produces": craft_times * recipe.output_count,
            "ingredients": {k: v * craft_times for k, v in recipe.ingredients.items()},
        })

        for ingredient, ingredient_count in recipe.ingredients.items():
            total_needed = ingredient_count * craft_times
            self._calculate_recursive(
                item=ingredient,
                count=total_needed,
                inventory=inventory,
                result=result,
                depth=depth + 1,
                max_depth=max_depth,
            )


def create_recipe_analyzer() -> RecipeDependencyAnalyzer:
    """Create a recipe dependency analyzer."""
    return RecipeDependencyAnalyzer(RECIPE_KNOWLEDGE)


def get_all_crafting_knowledge() -> List[KnowledgeItem]:
    """Get all crafting primitive knowledge items."""
    return CRAFTING_KNOWLEDGE.copy()
