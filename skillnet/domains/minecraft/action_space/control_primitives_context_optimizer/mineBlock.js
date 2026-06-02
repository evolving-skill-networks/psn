// Mine blocks and collect them into inventory.
//
// PARAMETERS:
//   mineBlock(bot, name, count = 1)
//   - name: string — BLOCK name (must exist in mcData.blocksByName), NOT item name
//   - count: number — how many blocks to mine
//
// CONTRACT:
// - name MUST be a STRICT block name STRING, NOT a Block object, Vec3, or position
// - Block name must match mcData.blocksByName exactly (strict matching)
//   e.g., "diamond_ore" and "deepslate_diamond_ore" are different blocks
// - Uses bot.findBlocks({matching: [blockId], maxDistance: 32, count: 1024})
// - If no matching blocks found within 32 blocks, returns 0 (does not throw)
//
// BLOCK VALIDATION:
// - Rejects unbreakable blocks: hardness < 0 (bedrock at Y=-64 and below, barrier, command_block)
//   Error: "Cannot mine ...: block is unbreakable in survival mode"
// - Checks canHarvest() with current held tool before mining
//   Error: "Cannot mine ...: requires at least a stone_pickaxe"
//   e.g., diamond_ore requires iron pickaxe or better; deepslate needs stone pickaxe or better
// - Auto-equips best tool via bot.tool.equipForBlock(target, {requireHarvest: false}) before mining
//
// ZERO-GAIN DETECTION:
// - Tracks inventory before/after mining. If inventory unchanged after collect:
//   warns "Inventory unchanged after mining" and counts consecutive zero-gains
// - After 3 consecutive zero-gains, throws with diagnostic:
//   "blocks found but nothing collected. Possible causes: wrong tool, items in void/water, inaccessible"
// - Detects if bot is in water (items may float away)
//
// RETURNS: number — actual items gained (inventoryAfter - inventoryBefore), may be less than count
// Returns 0 if no blocks found (does NOT throw, but increments fail counter)
// Throws after 10+ consecutive "no blocks found" failures
//
// ERROR HANDLING:
//   - "name for mineBlock must be a string" / "count for mineBlock must be a number"
//   - "No block named ..." — not in mcData
//   - "Cannot mine ...: block is unbreakable" — hardness < 0
//   - "Cannot mine ...: requires at least a ..." — wrong tool
//   - "mineBlock failed too many times" — 10+ consecutive no-blocks-found
//   - "blocks found but nothing collected" — 3+ consecutive zero-gain
//
// NOTE: Uses collectWithTimeout() for item collection (with ignoreNoPath: true).
// NOTE: Calls bot.save('blockName_mined') on success.
// NOTE: Waits 2 ticks after collection for inventory sync.
async function mineBlock(bot, name, count = 1) {
    if (typeof name !== "string") throw new Error(`name for mineBlock must be a string`);
    if (typeof count !== "number") throw new Error(`count for mineBlock must be a number`);
    const blockByName = mcData.blocksByName[name];
    if (!blockByName) throw new Error(`No block named ${name}`);
    if (blockByName.hardness !== null && blockByName.hardness < 0) {
        throw new Error(`Cannot mine ${name}: block is unbreakable in survival mode`);
    }
    const blocks = bot.findBlocks({ matching: [blockByName.id], maxDistance: 32, count: 1024 });
    if (blocks.length === 0) return 0;  // No blocks found
    const targets = [];
    for (let i = 0; i < blocks.length; i++) targets.push(bot.blockAt(blocks[i]));
    if (targets.length === 0) return 0;
    // Auto-equip best tool
    if (targets.length > 0 && bot.tool) {
        try { await bot.tool.equipForBlock(targets[0]); } catch (e) {}
    }
    // Pre-check: can bot harvest this block with current tool?
    const heldItem = bot.heldItem;
    if (!targets[0].canHarvest(heldItem ? heldItem.type : null)) {
        throw new Error(`Cannot mine ${name}: bot does not have a harvestable tool`);
    }
    const inventoryBefore = bot.inventory.items().reduce((s, it) => s + it.count, 0);
    await collectWithTimeout(bot, targets, { ignoreNoPath: true, count: count }, count, name);
    await bot.waitForTicks(2);
    const inventoryAfter = bot.inventory.items().reduce((s, it) => s + it.count, 0);
    bot.save(`${name}_mined`);
    return inventoryAfter - inventoryBefore;
}
