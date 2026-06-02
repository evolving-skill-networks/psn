// Craft items using bot.checkRecipe() for recipe validation + bot.craft() for execution.
// This function handles both 2x2 (no crafting table) and 3x3 (needs crafting table) recipes.
//
// PARAMETERS:
//   craftItem(bot, name, count = 1)
//   - name: string — item name (must exist in mcData.itemsByName)
//   - count: number — number of RECIPE EXECUTIONS (not output items)
//
// IMPORTANT: count = number of RECIPE EXECUTIONS, NOT the number of output items you want.
// Each recipe execution consumes inputs and produces a fixed number of outputs.
// Check the recipe output quantity before choosing count.
//
// RECIPE RESOLUTION (recipe checking is internal — do NOT call bot.recipesFor() yourself):
// - Finds placed crafting_table within 32 blocks via bot.findBlock()
// - Calls bot.checkRecipe(name, count, craftingTable) → {available, message, recipe}
// - 3x3 recipes (pickaxe, sword, furnace) REQUIRE a placed crafting_table nearby
// - 2x2 recipes (planks, sticks) work WITHOUT a crafting table
// - craftItem ONLY SEARCHES for an already-placed crafting table — it does NOT place one for you
// - You MUST ensure a crafting table block is placed in the world before calling craftItem for 3x3 recipes
//
// CRAFTING STRATEGY: Crafts one-at-a-time in a loop (avoids mineflayer inventory sync issues).
// Waits 2 ticks between each craft for inventory sync.
//
// ERROR HANDLING: craftItem THROWS on ALL failure cases:
//   - "name for craftItem must be a string" / "count for craftItem must be a number"
//   - "No item named ..." — item not in mcData
//   - "craftItem failed for ...: Insufficient materials..." — Missing materials for recipe
//   - "craftItem failed for ...: No recipe found..." — no recipe exists
//   - "... requires a crafting table (3x3 grid)" — 3x3 recipe but no table nearby
//   - "craftItem failed too many times" — 10+ consecutive failures (global counter)
// Do NOT suppress these errors with try-catch.
// WRONG: try { await craftItem(...) } catch(e) { bot.chat(e.message); return; }
// RIGHT: await craftItem(bot, "item_name", count);  // let errors propagate naturally
// Do NOT add redundant recipe checks before calling craftItem (but DO ensure crafting table is placed for 3x3 recipes).
//
// NOTE: Uses gotoWithTimeout(bot, goal, 30000) to reach crafting table (30s timeout).
// NOTE: Calls failedCraftFeedback() on insufficient materials for detailed chat feedback.
// NOTE: craftItem does NOT check if you already have the output — it always executes.
async function craftItem(bot, name, count = 1) {
    if (typeof name !== "string") throw new Error("name for craftItem must be a string");
    if (typeof count !== "number") throw new Error("count for craftItem must be a number");
    const itemByName = mcData.itemsByName[name];
    if (!itemByName) throw new Error(`No item named ${name}`);
    await bot.waitForTicks(2);
    const craftingTable = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 32,
    });
    if (craftingTable) {
        await gotoWithTimeout(bot, new GoalLookAtBlock(craftingTable.position, bot.world), 30000, 'crafting table');
    }
    const checkResult = bot.checkRecipe(name, count, craftingTable);
    if (!checkResult.available) {
        if (checkResult.reason === 'insufficient_materials') {
            failedCraftFeedback(bot, name, itemByName, craftingTable);
        }
        throw new Error(`craftItem failed for ${name}: ${checkResult.message}`);
    }
    for (let i = 0; i < count; i++) {
        const singleCheck = bot.checkRecipe(name, 1, craftingTable);
        if (!singleCheck.available) break;
        await bot.craft(singleCheck.recipe, 1, craftingTable);
        await bot.waitForTicks(2);
    }
}
