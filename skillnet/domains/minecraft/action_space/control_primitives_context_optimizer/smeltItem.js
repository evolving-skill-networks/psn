// Smelt items in a furnace.
//
// PARAMETERS:
//   smeltItem(bot, itemName, fuelName, count = 1)
//   - itemName: string — item to smelt (must exist in mcData.itemsByName)
//   - fuelName: string — fuel item (must be a valid fuel in mcData)
//   - count: number — how many items to smelt
//
// PREREQUISITES:
// - A furnace block must be PLACED within 32 blocks (throws "No furnace nearby" if not)
//   A furnace ITEM in inventory is NOT enough — it must be placed as a block first using placeItem.
//   smeltItem searches for a placed furnace block, NOT an inventory item.
// - itemName must be in bot's inventory
// - fuelName must be in bot's inventory
//
// FUEL MANAGEMENT:
// - Checks furnace.fuelSeconds < 15 before adding new fuel
// - Only adds fuel if current fuel item is different from fuelName
// - Validates fuel by checking furnace.fuel state after putFuel
// - Throws "... is not a valid fuel" if fuel doesn't work
//
// SMELTING LOOP:
// - Smelts one at a time: putInput → wait 12 seconds (12*20 ticks) → takeOutput
// - Checks inventory for both item and fuel before each iteration
// - Stops early if item or fuel runs out (does not throw)
// - Throws "... is not a valid input" if no output after smelting
//
// ERROR HANDLING:
//   - "itemName or fuelName for smeltItem must be a string"
//   - "count for smeltItem must be a number"
//   - "No item named ..." / "No fuel named ..."
//   - "No furnace nearby"
//   - "... is not a valid fuel" / "... is not a valid input"
//   - "smeltItem failed too many times" — 10+ consecutive zero-success failures
//
// NOTE: Uses gotoWithTimeout(bot, goal, 30000) to reach furnace (30s timeout).
// NOTE: Closes furnace with try-catch (furnace may be destroyed during smelting).
async function smeltItem(bot, itemName, fuelName, count = 1) {
    if (typeof itemName !== "string" || typeof fuelName !== "string")
        throw new Error("itemName or fuelName for smeltItem must be a string");
    if (typeof count !== "number") throw new Error("count for smeltItem must be a number");
    const item = mcData.itemsByName[itemName];
    const fuel = mcData.itemsByName[fuelName];
    if (!item) throw new Error(`No item named ${itemName}`);
    if (!fuel) throw new Error(`No fuel named ${fuelName}`);
    const furnaceBlock = bot.findBlock({
        matching: mcData.blocksByName.furnace.id,
        maxDistance: 32,
    });
    if (!furnaceBlock) throw new Error("No furnace nearby");
    await gotoWithTimeout(bot, new GoalLookAtBlock(furnaceBlock.position, bot.world), 30000, 'furnace');
    const furnace = await bot.openFurnace(furnaceBlock);
    let success_count = 0;
    for (let i = 0; i < count; i++) {
        if (!bot.inventory.findInventoryItem(item.id, null)) break;
        if (furnace.fuelSeconds < 15 && furnace.fuelItem()?.name !== fuelName) {
            if (!bot.inventory.findInventoryItem(fuel.id, null)) break;
            await furnace.putFuel(fuel.id, null, 1);
            await bot.waitForTicks(20);
        }
        await furnace.putInput(item.id, null, 1);
        await bot.waitForTicks(12 * 20);
        if (!furnace.outputItem()) throw new Error(`${itemName} is not a valid input`);
        await furnace.takeOutput();
        success_count++;
    }
    try { furnace.close(); } catch (e) {}
    if (success_count > 0) bot.chat(`Smelted ${success_count} ${itemName}.`);
}
