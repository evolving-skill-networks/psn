async function mineBlock(bot, name, count = 1) {
    // CONTRACT:
    // - name MUST be a block name STRING (e.g., "lapis_ore"), NOT a Block object, Vec3, position, or any other object.
    // - count MUST be a number.
    // return if name is not string
    if (typeof name !== "string") {
        throw new Error(`name for mineBlock must be a string`);
    }
    if (typeof count !== "number") {
        throw new Error(`count for mineBlock must be a number`);
    }
    const blockByName = mcData.blocksByName[name];
    if (!blockByName) {
        throw new Error(`No block named ${name}`);
    }
    // Reject unbreakable blocks (hardness < 0, e.g. bedrock, barrier, command_block)
    if (blockByName.hardness !== null && blockByName.hardness < 0) {
        throw new Error(
            `Cannot mine ${name}: block is unbreakable in survival mode (hardness=${blockByName.hardness}). ` +
            `Blocks with negative hardness (bedrock at Y<=-63, barrier, etc.) cannot be broken by any tool.`
        );
    }
    const blocks = bot.findBlocks({
        matching: [blockByName.id],
        maxDistance: 32,
        count: 1024,
    });
    if (blocks.length === 0) {
        bot.chat(`No ${name} nearby, please explore first`);
        _mineBlockFailCount++;
        if (_mineBlockFailCount > 10) {
            throw new Error(
                "mineBlock failed too many times, make sure you explore before calling mineBlock"
            );
        }
        return 0;
    }
    const targets = [];
    for (let i = 0; i < blocks.length; i++) {
        targets.push(bot.blockAt(blocks[i]));
    }

    // Explicitly equip the best tool before mining
    // This ensures tools are used even if collectBlock's auto-equip fails
    if (targets.length > 0 && bot.tool) {
        try {
            await bot.tool.equipForBlock(targets[0], { requireHarvest: false });
        } catch (e) {
            // Ignore equip errors - mining can still proceed with current tool
        }
    }

    // Pre-check: can the bot actually harvest this block?
    // Prevents infinite loops where mineBlock returns 0 silently because the bot
    // lacks the required tool (e.g., mining deepslate without a pickaxe).
    const heldItem = bot.heldItem;
    if (!targets[0].canHarvest(heldItem ? heldItem.type : null)) {
        const properties = mcData.blocksByName[name];
        if (properties && properties.harvestTools) {
            const toolId = Object.keys(properties.harvestTools)[0];
            const toolItem = mcData.items[toolId];
            throw new Error(
                `Cannot mine ${name}: requires at least a ${toolItem ? toolItem.name : 'proper tool'}. ` +
                `Craft or find the required tool first.`
            );
        }
        throw new Error(`Cannot mine ${name}: bot does not have a harvestable tool.`);
    }

    // P1: inventory change detection - detect "fake success" loops (Collect finish! but inventory unchanged)
    // Record inventory count before mining
    const getInventoryCount = () => {
        let total = 0;
        for (const item of bot.inventory.items()) {
            total += item.count;
        }
        return total;
    };
    const inventoryBefore = getInventoryCount();

    await collectWithTimeout(bot, targets, {
        ignoreNoPath: true,
        count: count,
    }, count, name);

    // Wait for inventory sync to complete (so subsequent operations see the latest inventory state)
    await bot.waitForTicks(2);

    // P1: check whether inventory has changed
    const inventoryAfter = getInventoryCount();
    if (inventoryAfter === inventoryBefore) {
        _mineBlockZeroGainCount++;
        console.warn(`[mineBlock] Warning: Inventory unchanged after mining ${name} (consecutive: ${_mineBlockZeroGainCount}). Bot may be stuck or items unreachable.`);
        // Check whether bot is in water
        if (bot.entity) {
            const blockAtBot = bot.blockAt(bot.entity.position);
            if (blockAtBot && blockAtBot.name.includes('water')) {
                console.warn(`[mineBlock] Bot is in water at ${bot.entity.position}. Items may have fallen into water.`);
                bot.chat(`I'm stuck in water and can't collect items. Need to move to a different location.`);
            }
        }
        if (_mineBlockZeroGainCount > 3) {
            _mineBlockZeroGainCount = 0;
            throw new Error(
                `mineBlock failed 3 consecutive times for ${name}: blocks found but nothing collected. ` +
                `Possible causes: wrong tool, items falling into void/water, or blocks are inaccessible.`
            );
        }
    } else {
        _mineBlockZeroGainCount = 0;
    }

    bot.save(`${name}_mined`);
    // Return actual items gained (may be less than count if blocks unreachable)
    return inventoryAfter - inventoryBefore;
}
