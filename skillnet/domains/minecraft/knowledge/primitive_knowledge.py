"""
Primitive Knowledge - precondition and failure-mode knowledge for control primitives.

When an error occurs in a primitive call (e.g. craftItem, mineBlock) rather than
a skill call, this module provides diagnostic information and fix suggestions.

Core features:
1. Define the preconditions for each primitive
2. Match common failure patterns
3. Generate concrete fix strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import re


@dataclass
class PrimitiveFailurePattern:
    """
    Primitive failure pattern.

    v7.0 refactor: removed fix_hint/fix_strategy fields, keeping only a purely
    factual description. The LLM should reason about the fix itself based on cause.
    """
    pattern: str  # Error message matching pattern (regular expression)
    cause: str  # Failure cause (purely factual description)


@dataclass
class PrimitivePrecondition:
    """
    Primitive precondition.

    v7.0 refactor: removed the check_hint field, keeping only a purely factual description.
    """
    description: str  # Precondition description (purely factual)


@dataclass
class PrimitiveKnowledge:
    """Knowledge for a single primitive."""
    name: str
    description: str
    preconditions: List[PrimitivePrecondition] = field(default_factory=list)
    failure_patterns: List[PrimitiveFailurePattern] = field(default_factory=list)
    common_parameters: List[str] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)  # Related ensure/wrapper skills


# ============================================================
# Primitive knowledge base definitions
# ============================================================

PRIMITIVE_KNOWLEDGE: Dict[str, PrimitiveKnowledge] = {
    # --------------------------------------------------------
    # craftItem - craft an item
    # --------------------------------------------------------
    "craftItem": PrimitiveKnowledge(
        name="craftItem",
        description="Crafts an item using available materials in inventory",
        preconditions=[
            PrimitivePrecondition(
                description="Must have sufficient raw materials in inventory",
            ),
            PrimitivePrecondition(
                description="For 3x3 recipes, a crafting table must be nearby (within 32 blocks)",
            ),
            PrimitivePrecondition(
                description="Recipe must exist for the target item",
            ),
        ],
        failure_patterns=[
            PrimitiveFailurePattern(
                pattern=r"[Ii]nsufficient materials?|[Mm]issing:?\s*\d*\s*\w+|[Nn]ot enough",
                cause="Inventory lacks required crafting materials",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Nn]o crafting table|[Nn]eed.*crafting table|3x3 recipe",
                cause="Complex recipe requires crafting table but none nearby",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Nn]o item named|[Nn]o such item|[Ii]nvalid item",
                cause="Item name is incorrect or doesn't exist",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Rr]ecipe not found|[Cc]annot craft|[Nn]o recipe",
                cause="No crafting recipe exists for this item",
            ),
        ],
        common_parameters=["bot", "name", "count"],
        related_skills=["ensureItem", "ensureCraftingTable", "gatherMaterials"],
    ),

    # --------------------------------------------------------
    # mineBlock - mine a block
    # --------------------------------------------------------
    "mineBlock": PrimitiveKnowledge(
        name="mineBlock",
        description="Mines/breaks a block and collects drops",
        preconditions=[
            PrimitivePrecondition(
                description="Block must exist and be reachable",
            ),
            PrimitivePrecondition(
                description="Must have appropriate tool equipped for efficient mining",
            ),
            PrimitivePrecondition(
                description="Must be within interaction range (4 blocks)",
            ),
        ],
        failure_patterns=[
            PrimitiveFailurePattern(
                pattern=r"[Nn]o ?\w* ?found nearby|[Cc]annot find|[Bb]lock not visible|no ore found",
                cause="Target block doesn't exist in search range",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Dd]ig failed|[Cc]annot break|[Ww]rong tool|[Tt]ool required",
                cause="Wrong or no tool equipped for this block type",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Pp]ath.*blocked|[Cc]annot reach|[Uu]nreachable|[Tt]imeout",
                cause="Cannot pathfind to block location",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Ii]nventory full|[Nn]o space|[Cc]annot pick up",
                cause="Inventory full, cannot collect drops",
            ),
        ],
        common_parameters=["bot", "name", "count"],
        related_skills=["ensurePickaxe", "exploreUntil", "mineOre"],
    ),

    # --------------------------------------------------------
    # placeItem - place an item
    # --------------------------------------------------------
    "placeItem": PrimitiveKnowledge(
        name="placeItem",
        description="Places an item from inventory as a block in the world",
        preconditions=[
            PrimitivePrecondition(
                description="Item must be in inventory",
            ),
            PrimitivePrecondition(
                description="Valid placement position with adjacent solid block",
            ),
            PrimitivePrecondition(
                description="Position must not be occupied by another block/entity",
            ),
        ],
        failure_patterns=[
            PrimitiveFailurePattern(
                pattern=r"[Nn]o valid.*position|[Cc]annot place|[Pp]lacement failed|[Nn]o block to place on",
                cause="No suitable surface to place block against",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Bb]lockUpdate.*timeout|did not fire",
                cause="Block placed but not confirmed, possible position conflict",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Dd]on't have|[Nn]ot in inventory|[Mm]issing item",
                cause="Item to place not in inventory",
            ),
        ],
        common_parameters=["bot", "name", "position"],
        related_skills=["ensureCraftingTable", "placeFurnace", "buildPlatform"],
    ),

    # --------------------------------------------------------
    # smeltItem - smelt an item
    # --------------------------------------------------------
    "smeltItem": PrimitiveKnowledge(
        name="smeltItem",
        description="Smelts items in a furnace",
        preconditions=[
            PrimitivePrecondition(
                description="Furnace must be placed and accessible",
            ),
            PrimitivePrecondition(
                description="Must have items to smelt in inventory",
            ),
            PrimitivePrecondition(
                description="Must have fuel in inventory",
            ),
        ],
        failure_patterns=[
            PrimitiveFailurePattern(
                pattern=r"[Nn]o furnace|[Ff]urnace not found|[Nn]eed furnace",
                cause="No furnace nearby or accessible",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Nn]o fuel|[Oo]ut of fuel|[Ff]uel required",
                cause="No fuel available for smelting",
            ),
        ],
        common_parameters=["bot", "name", "count"],
        related_skills=["ensureFurnace", "ensureCoal", "smeltRawIron"],
    ),

    # --------------------------------------------------------
    # killMob - kill a mob
    # --------------------------------------------------------
    "killMob": PrimitiveKnowledge(
        name="killMob",
        description="Attacks and kills a mob entity",
        preconditions=[
            PrimitivePrecondition(
                description="Target mob must exist and be visible",
            ),
            PrimitivePrecondition(
                description="Must be within attack range",
            ),
            PrimitivePrecondition(
                description="Preferably have weapon equipped",
            ),
        ],
        failure_patterns=[
            PrimitiveFailurePattern(
                pattern=r"[Ee]ntity not found|[Tt]arget.*not found|[Nn]o nearby|[Nn]o \w+ found",
                cause="Target mob doesn't exist in range",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Oo]ut of range|[Tt]oo far|[Cc]annot reach",
                cause="Mob is too far to attack",
            ),
        ],
        common_parameters=["bot", "mobName", "count"],
        related_skills=["findAndApproachEntity", "ensureSword", "huntAnimal"],
    ),

    # --------------------------------------------------------
    # gotoWithTimeout - pathfind to a location
    # --------------------------------------------------------
    "gotoWithTimeout": PrimitiveKnowledge(
        name="gotoWithTimeout",
        description="Pathfinds to a location with timeout",
        preconditions=[
            PrimitivePrecondition(
                description="Destination must be reachable (not blocked by walls/water)",
            ),
        ],
        failure_patterns=[
            PrimitiveFailurePattern(
                pattern=r"[Pp]ath.*timeout|[Pp]athfinder timeout|[Cc]ould not find path",
                cause="Pathfinding couldn't find route in time limit",
            ),
            PrimitiveFailurePattern(
                pattern=r"[Gg]oal.*unreachable|[Cc]annot reach|[Bb]locked",
                cause="Path is completely blocked",
            ),
        ],
        common_parameters=["bot", "goal", "timeout"],
        related_skills=["exploreUntil", "digToDestination", "bridgeAcross"],
    ),
}


class PrimitiveKnowledgeBase:
    """
    Primitive knowledge-base manager.

    Provides query and matching functionality.
    """

    def __init__(
        self,
        knowledge: Optional[Dict[str, PrimitiveKnowledge]] = None,
        recipe_analyzer: Optional['RecipeDependencyAnalyzer'] = None,
    ):
        self.knowledge = knowledge or PRIMITIVE_KNOWLEDGE
        self._recipe_analyzer = recipe_analyzer

    @property
    def recipe_analyzer(self) -> 'RecipeDependencyAnalyzer':
        """Get the recipe analyzer (lazy-loaded)."""
        if self._recipe_analyzer is None:
            self._recipe_analyzer = RecipeDependencyAnalyzer(RECIPE_KNOWLEDGE)
        return self._recipe_analyzer

    def get_primitive(self, name: str) -> Optional[PrimitiveKnowledge]:
        """Get knowledge for the specified primitive."""
        return self.knowledge.get(name)

    def detect_primitive_in_error(self, error_message: str) -> List[str]:
        """Detect primitives involved in the error message."""
        detected = []
        error_lower = error_message.lower()

        # Direct primitive-name matches
        for name in self.knowledge.keys():
            if name.lower() in error_lower:
                detected.append(name)

        # Keyword matching
        keyword_mapping = {
            "craftItem": ["craft", "crafting", "recipe", "ingredients", "materials"],
            "mineBlock": ["mine", "mining", "dig", "break", "block"],
            "placeItem": ["place", "placement", "placing", "put"],
            "smeltItem": ["smelt", "smelting", "furnace", "fuel"],
            "killMob": ["kill", "attack", "combat", "fight", "entity", "mob"],
            "gotoWithTimeout": ["path", "goto", "reach", "navigate", "move to"],
        }

        for primitive, keywords in keyword_mapping.items():
            if primitive not in detected:
                if any(kw in error_lower for kw in keywords):
                    detected.append(primitive)

        return detected

    def match_failure_pattern(
        self,
        primitive_name: str,
        error_message: str,
    ) -> Optional[PrimitiveFailurePattern]:
        """Match the failure pattern."""
        primitive = self.get_primitive(primitive_name)
        if not primitive:
            return None

        for pattern in primitive.failure_patterns:
            if re.search(pattern.pattern, error_message, re.IGNORECASE):
                return pattern

        return None

    def analyze_error(
        self,
        error_message: str,
        skill_code: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze the error and return diagnostic information and fix suggestions.

        Args:
            error_message: Error message
            skill_code: Optional skill code, for more precise analysis

        Returns:
            Dict containing:
            - primitives: list of primitives involved
            - matched_patterns: matched failure patterns
            - precondition_hints: precondition check hints
            - fix_strategies: fix strategies
            - related_skills: related helper skills
        """
        result = {
            "primitives": [],
            "matched_patterns": [],
            "precondition_hints": [],
            "fix_strategies": [],
            "related_skills": set(),
        }

        # Detect involved primitives
        detected = self.detect_primitive_in_error(error_message)

        # If code is provided, also extract primitive calls from the code
        if skill_code:
            for name in self.knowledge.keys():
                if name in skill_code:
                    if name not in detected:
                        detected.append(name)

        result["primitives"] = detected

        # Analyze each primitive
        for prim_name in detected:
            primitive = self.get_primitive(prim_name)
            if not primitive:
                continue

            # Match the failure pattern
            pattern = self.match_failure_pattern(prim_name, error_message)
            if pattern:
                result["matched_patterns"].append({
                    "primitive": prim_name,
                    "cause": pattern.cause,
                })

            # Add precondition hints
            for precond in primitive.preconditions:
                result["precondition_hints"].append({
                    "primitive": prim_name,
                    "description": precond.description,
                })

            # Collect related skills
            result["related_skills"].update(primitive.related_skills)

        result["related_skills"] = list(result["related_skills"])
        return result

    def generate_fix_prompt(
        self,
        error_message: str,
        skill_code: str = "",
    ) -> str:
        """
        Generate a fix prompt for the LLM.

        Args:
            error_message: Error message
            skill_code: Skill code

        Returns:
            str: Formatted fix prompt
        """
        analysis = self.analyze_error(error_message, skill_code)

        if not analysis["primitives"]:
            return ""

        lines = ["## Primitive Failure Analysis\n"]

        # Primitives involved
        lines.append(f"**Primitives involved**: {', '.join(analysis['primitives'])}\n")

        # Matched patterns (only output cause, not fix_hint)
        if analysis["matched_patterns"]:
            lines.append("### Matched Failure Patterns:\n")
            for mp in analysis["matched_patterns"]:
                lines.append(f"- **{mp['primitive']}**: {mp['cause']}\n")

        # If it is an insufficient-materials error, add recipe-calculation help
        if self._is_material_error(error_message):
            recipe_help = self._generate_recipe_help(error_message, skill_code)
            if recipe_help:
                lines.append(recipe_help)

        # Preconditions (only output the description, not check_hint)
        if analysis["precondition_hints"]:
            lines.append("### Preconditions:\n")
            for pc in analysis["precondition_hints"][:5]:  # Limit count
                lines.append(f"- [{pc['primitive']}] {pc['description']}\n")

        # NOTE: removed the fix_strategies section to let the LLM reason on its own

        # NOTE: intentionally NO "Related Skills to Consider" section here.
        # It used to render analysis["related_skills"], which is a static,
        # hard-coded wishlist of aspirational skill names (e.g.
        # ensureCraftingTable / ensureItem / gatherMaterials) that are not
        # guaranteed to exist in the graph. Naming a non-existent skill drove
        # the optimizer to inline it (-> Responsibility Check reject) or call a
        # phantom (-> Reference Check reject), a deadlock that burned craftAxe's
        # optimization rounds in the qwen3-coder-next e2e run (round 0 generated
        # `async function ensureCraftingTable` inline). The real, graph-derived
        # skill list is injected separately by Phase 1 as the "Available learned
        # skills" composable section (pure_reflection.py), which only lists
        # skills that actually exist. Keep the factual primitive/precondition
        # knowledge above; let that composable section be the single source of
        # callable skill names.

        return "\n".join(lines)

    def _is_material_error(self, error_message: str) -> bool:
        """Check whether this is an insufficient-materials error."""
        patterns = [
            r"[Ii]nsufficient materials?",
            r"[Mm]issing:?\s*\d*\s*\w+",
            r"[Nn]ot enough",
            r"[Nn]eed \d+ more",
        ]
        return any(re.search(p, error_message) for p in patterns)

    def _generate_recipe_help(self, error_message: str, skill_code: str) -> str:
        """Generate recipe-calculation help."""
        lines = ["### Recipe Dependency Analysis\n"]

        # Extract item names from the error message
        # For example: "Missing: 2 oak_planks" -> "oak_planks"
        item_match = re.search(
            r"[Mm]issing:?\s*\d*\s*(\w+)|"
            r"[Nn]eed \d+ more (\w+)|"
            r"craft (\w+).*[Ff]ailed|"
            r"[Ii]nsufficient.*craft (\w+)",
            error_message
        )

        mentioned_items = []
        if item_match:
            for g in item_match.groups():
                if g:
                    mentioned_items.append(g)

        # Extract possible target items from the code
        code_items = []
        if skill_code:
            # Match craftItem(bot, "item_name", count)
            for m in re.finditer(r'craftItem\s*\(\s*bot\s*,\s*["\'](\w+)["\']', skill_code):
                code_items.append(m.group(1))

        # Merge and deduplicate
        all_items = list(set(mentioned_items + code_items))

        if not all_items:
            lines.append("Could not identify target items from error message.\n")
            lines.append("**General Recipe Calculation Tips:**\n")
        else:
            lines.append(f"**Items mentioned**: {', '.join(all_items)}\n")

        # Generate recipe info for each item
        for item in all_items[:3]:  # Limit count
            recipe = self.recipe_analyzer.get_recipe(item)
            if recipe:
                lines.append(f"\n**Recipe for {item}:**")
                lines.append(f"  - Produces: {recipe.output_count} per craft")
                lines.append(f"  - Ingredients: {', '.join(f'{v}x {k}' for k, v in recipe.ingredients.items())}")
                lines.append(f"  - Needs crafting table: {recipe.requires_table}")

                # Compute the complete dependency chain
                calc = self.recipe_analyzer.calculate_materials(item, 1)
                if calc.total_raw_materials:
                    lines.append(f"  - **Total raw materials for 1x {item}**: " +
                                ", ".join(f"{v}x {k}" for k, v in calc.total_raw_materials.items()))

        # Add general tips
        lines.append("\n### CRITICAL: Material Calculation Rules\n")
        lines.append("**Problem**: Intermediate crafting consumes materials that may also be needed elsewhere.")
        lines.append("")
        lines.append("**Example - wooden_pickaxe calculation error**:")
        lines.append("```")
        lines.append("WRONG: need 3 planks (for pickaxe) + check if have 2 sticks")
        lines.append("RIGHT: need 3 planks (for pickaxe) + 2 planks (for sticks) = 5 planks total")
        lines.append("```")
        lines.append("")
        lines.append("**Correct approach**:")
        lines.append("1. Calculate ALL materials needed BEFORE any crafting")
        lines.append("2. Include dependencies: if sticks need planks, add those planks")
        lines.append("3. Use recipe output counts: planks give 4, sticks give 4")
        lines.append("4. Formula: `delta = ceil(needed / output_count) * ingredient_count - have`")

        return "\n".join(lines)


# ============================================================
# Recipe dependency-chain knowledge - solves material calculation errors
# ============================================================

@dataclass
class RecipeInfo:
    """Recipe information."""
    output: str  # Output item
    output_count: int  # Quantity produced per craft
    ingredients: Dict[str, int]  # Raw material -> required quantity
    requires_table: bool = False  # Whether a crafting table is required


# Minecraft core recipe library
RECIPE_KNOWLEDGE: Dict[str, RecipeInfo] = {
    # Wood products
    "oak_planks": RecipeInfo(
        output="oak_planks",
        output_count=4,
        ingredients={"oak_log": 1},
        requires_table=False,
    ),
    "birch_planks": RecipeInfo(
        output="birch_planks",
        output_count=4,
        ingredients={"birch_log": 1},
        requires_table=False,
    ),
    "spruce_planks": RecipeInfo(
        output="spruce_planks",
        output_count=4,
        ingredients={"spruce_log": 1},
        requires_table=False,
    ),
    "stick": RecipeInfo(
        output="stick",
        output_count=4,
        ingredients={"oak_planks": 2},  # Or any plank type
        requires_table=False,
    ),
    "crafting_table": RecipeInfo(
        output="crafting_table",
        output_count=1,
        ingredients={"oak_planks": 4},
        requires_table=False,
    ),

    # Tools
    "wooden_pickaxe": RecipeInfo(
        output="wooden_pickaxe",
        output_count=1,
        ingredients={"oak_planks": 3, "stick": 2},
        requires_table=True,
    ),
    "stone_pickaxe": RecipeInfo(
        output="stone_pickaxe",
        output_count=1,
        ingredients={"cobblestone": 3, "stick": 2},
        requires_table=True,
    ),
    "iron_pickaxe": RecipeInfo(
        output="iron_pickaxe",
        output_count=1,
        ingredients={"iron_ingot": 3, "stick": 2},
        requires_table=True,
    ),
    "wooden_axe": RecipeInfo(
        output="wooden_axe",
        output_count=1,
        ingredients={"oak_planks": 3, "stick": 2},
        requires_table=True,
    ),
    "wooden_shovel": RecipeInfo(
        output="wooden_shovel",
        output_count=1,
        ingredients={"oak_planks": 1, "stick": 2},
        requires_table=True,
    ),
    "wooden_sword": RecipeInfo(
        output="wooden_sword",
        output_count=1,
        ingredients={"oak_planks": 2, "stick": 1},
        requires_table=True,
    ),

    # Furnace-related
    "furnace": RecipeInfo(
        output="furnace",
        output_count=1,
        ingredients={"cobblestone": 8},
        requires_table=True,
    ),
    "iron_ingot": RecipeInfo(
        output="iron_ingot",
        output_count=1,
        ingredients={"raw_iron": 1},  # Requires smelting
        requires_table=False,
    ),

    # Others
    "torch": RecipeInfo(
        output="torch",
        output_count=4,
        ingredients={"coal": 1, "stick": 1},
        requires_table=False,
    ),
    "chest": RecipeInfo(
        output="chest",
        output_count=1,
        ingredients={"oak_planks": 8},
        requires_table=True,
    ),
    "boat": RecipeInfo(
        output="boat",
        output_count=1,
        ingredients={"oak_planks": 5},
        requires_table=True,
    ),
}


@dataclass
class MaterialCalculation:
    """Material calculation result."""
    target_item: str
    target_count: int
    total_raw_materials: Dict[str, int]  # Final raw materials needed
    intermediate_crafts: List[Dict[str, Any]]  # Intermediate crafting steps
    warnings: List[str]  # Calculation warnings


class RecipeDependencyAnalyzer:
    """
    Recipe dependency-chain analyzer.

    Problem solved: when crafting a wooden_pickaxe, you need:
    - 3 planks (directly)
    - 2 sticks (directly)
    - But sticks require 2 planks to craft!

    So actually you need: 3 + 2 = 5 planks (if you have no sticks).

    This analyzer can:
    1. Compute the full dependency chain
    2. Consider materials already in inventory
    3. Generate the correct material requirements list
    """

    def __init__(self, recipes: Optional[Dict[str, RecipeInfo]] = None):
        self.recipes = recipes or RECIPE_KNOWLEDGE

    def get_recipe(self, item: str) -> Optional[RecipeInfo]:
        """Get the recipe."""
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
            target_item: Target item
            target_count: Required quantity
            inventory: Current inventory {item: count}
            max_depth: Maximum recursion depth

        Returns:
            MaterialCalculation: Complete material-calculation result
        """
        inventory = inventory or {}
        result = MaterialCalculation(
            target_item=target_item,
            target_count=target_count,
            total_raw_materials={},
            intermediate_crafts=[],
            warnings=[],
        )

        # Recursive calculation
        self._calculate_recursive(
            item=target_item,
            count=target_count,
            inventory=dict(inventory),  # Copy to avoid modifying the original
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

        # First check inventory
        have = inventory.get(item, 0)
        if have >= count:
            # Inventory is sufficient; consume it
            inventory[item] = have - count
            return

        # Quantity that needs to be crafted
        need_to_craft = count - have
        inventory[item] = 0  # Inventory exhausted

        # Look up the recipe
        recipe = self.get_recipe(item)
        if not recipe:
            # No recipe: this is a raw material
            result.total_raw_materials[item] = (
                result.total_raw_materials.get(item, 0) + need_to_craft
            )
            return

        # Compute how many times to craft
        craft_times = (need_to_craft + recipe.output_count - 1) // recipe.output_count

        # Record the intermediate step
        result.intermediate_crafts.append({
            "item": item,
            "craft_times": craft_times,
            "produces": craft_times * recipe.output_count,
            "ingredients": {k: v * craft_times for k, v in recipe.ingredients.items()},
        })

        # Recursively process each ingredient
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

    def generate_calculation_prompt(
        self,
        target_item: str,
        target_count: int = 1,
        inventory: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Generate a material-calculation prompt for the LLM.

        Args:
            target_item: Target item
            target_count: Required quantity
            inventory: Current inventory

        Returns:
            str: Formatted prompt
        """
        calc = self.calculate_materials(target_item, target_count, inventory)

        lines = [f"## Material Calculation for {target_item} x{target_count}\n"]

        if inventory:
            lines.append("### Current Inventory:")
            for item, count in inventory.items():
                lines.append(f"  - {item}: {count}")
            lines.append("")

        if calc.intermediate_crafts:
            lines.append("### Crafting Steps (in order):")
            for i, step in enumerate(reversed(calc.intermediate_crafts), 1):
                lines.append(
                    f"{i}. Craft {step['item']} x{step['craft_times']} "
                    f"(produces {step['produces']})"
                )
                for ing, cnt in step['ingredients'].items():
                    lines.append(f"   - needs {ing} x{cnt}")
            lines.append("")

        if calc.total_raw_materials:
            lines.append("### Raw Materials Needed:")
            for item, count in calc.total_raw_materials.items():
                lines.append(f"  - {item}: {count}")
            lines.append("")

        if calc.warnings:
            lines.append("### Warnings:")
            for w in calc.warnings:
                lines.append(f"  - {w}")

        # Add hints about common mistakes
        lines.append("### Common Mistakes to Avoid:")
        lines.append("1. **Don't forget intermediate materials consume resources**")
        lines.append("   - Example: sticks need planks, so wooden_pickaxe needs 3+2=5 planks total")
        lines.append("2. **Calculate delta correctly**")
        lines.append("   - delta = total_needed - inventory_count")
        lines.append("3. **Account for recipe output counts**")
        lines.append("   - Crafting planks gives 4, sticks gives 4, etc.")

        return "\n".join(lines)


def create_recipe_analyzer() -> RecipeDependencyAnalyzer:
    """Create the recipe-dependency analyzer."""
    return RecipeDependencyAnalyzer(RECIPE_KNOWLEDGE)


def create_primitive_knowledge_base() -> PrimitiveKnowledgeBase:
    """Create the primitive knowledge-base instance."""
    return PrimitiveKnowledgeBase(PRIMITIVE_KNOWLEDGE)
