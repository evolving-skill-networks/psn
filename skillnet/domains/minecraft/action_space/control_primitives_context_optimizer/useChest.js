// Chest interaction functions: get items, deposit items, check contents.
//
// THREE MAIN FUNCTIONS:
//
// 1. getItemFromChest(bot, chestPosition, itemsToGet)
//    - chestPosition: Vec3 — chest block position
//    - itemsToGet: object — {itemName: count, ...}
//    - Withdraws items from chest. Skips missing items with chat message.
//
// 2. depositItemIntoChest(bot, chestPosition, itemsToDeposit)
//    - chestPosition: Vec3 — chest block position
//    - itemsToDeposit: object — {itemName: count, ...}
//    - Deposits items into chest. Skips missing items with chat message.
//
// 3. checkItemInsideChest(bot, chestPosition)
//    - chestPosition: Vec3 — chest block position
//    - Opens chest, lists contents, then closes. Returns nothing.
//
// HELPER FUNCTIONS (called internally):
//
// moveToChest(bot, chestPosition)
//   - Validates chestPosition is Vec3
//   - If distance > 32 blocks: teleports via /tp command
//   - Verifies block at position is actually "chest" (throws if not)
//   - Uses gotoWithTimeout(bot, goal, 30000) to navigate (30s timeout)
//
// listItemsInChest(bot, chestBlock)
//   - Opens container, aggregates items by name
//   - Emits "closeChest" event with item counts dict + position
//   - Returns the chest container object
//
// closeChest(bot, chestBlock)
//   - Calls listItemsInChest first (to emit inventory event)
//   - Then closes the container
//   - Falls back to bot.closeWindow() if listItemsInChest fails
//
// ERROR HANDLING:
//   - "chestPosition ... must be a Vec3" — for all 3 main functions
//   - "Bot entity is undefined" — bot disconnected
//   - "No chest at ..." — block at position is not a chest
//   - "No item named ..." — item not in mcData
//   - "I don't see ... in this chest" / "Not enough ... in chest" — chest doesn't have items
//   - "No ... in inventory" / "Not enough ... in inventory" — bot doesn't have items
//
// NOTE: All functions validate chestPosition is Vec3 instance.
// NOTE: Uses gotoWithTimeout for navigation, NOT plain pathfinder.goto().
async function getItemFromChest(bot, chestPosition, itemsToGet) {
    if (!(chestPosition instanceof Vec3)) throw new Error("chestPosition must be a Vec3");
    await moveToChest(bot, chestPosition);
    const chestBlock = bot.blockAt(chestPosition);
    const chest = await bot.openContainer(chestBlock);
    for (const name in itemsToGet) {
        const itemByName = mcData.itemsByName[name];
        if (!itemByName) { bot.chat(`No item named ${name}`); continue; }
        const item = chest.findContainerItem(itemByName.id);
        if (!item) { bot.chat(`I don't see ${name} in this chest`); continue; }
        try { await chest.withdraw(item.type, null, itemsToGet[name]); }
        catch (err) { bot.chat(`Not enough ${name} in chest.`); }
    }
    await closeChest(bot, chestBlock);
}

async function depositItemIntoChest(bot, chestPosition, itemsToDeposit) {
    if (!(chestPosition instanceof Vec3)) throw new Error("chestPosition must be a Vec3");
    await moveToChest(bot, chestPosition);
    const chestBlock = bot.blockAt(chestPosition);
    const chest = await bot.openContainer(chestBlock);
    for (const name in itemsToDeposit) {
        const itemByName = mcData.itemsByName[name];
        if (!itemByName) { bot.chat(`No item named ${name}`); continue; }
        const item = bot.inventory.findInventoryItem(itemByName.id);
        if (!item) { bot.chat(`No ${name} in inventory`); continue; }
        try { await chest.deposit(item.type, null, itemsToDeposit[name]); }
        catch (err) { bot.chat(`Not enough ${name} in inventory.`); }
    }
    await closeChest(bot, chestBlock);
}

async function checkItemInsideChest(bot, chestPosition) {
    if (!(chestPosition instanceof Vec3)) throw new Error("chestPosition must be a Vec3");
    await moveToChest(bot, chestPosition);
    const chestBlock = bot.blockAt(chestPosition);
    await bot.openContainer(chestBlock);
    await closeChest(bot, chestBlock);
}
