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
        // Carry the search radius and position so the failure is actionable
        // (which area was already searched) instead of a bare "nearby".
        const p = bot.entity ? bot.entity.position : null;
        const here = p
            ? ` within 32 blocks of (${Math.floor(p.x)}, ${Math.floor(p.y)}, ${Math.floor(p.z)})`
            : "";
        bot.chat(`No ${name} found${here}, please explore first`);
        _mineBlockFailCount++;
        if (_mineBlockFailCount > 10) {
            throw new Error(
                `mineBlock failed too many times: no ${name} found${here} after repeated ` +
                `searches. Explore a different area before calling mineBlock again.`
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
    // Count only the TARGET BLOCK'S OWN DROPS. A type-blind total let unrelated
    // pickups (cobble from the approach tunnel, decayed-leaf junk) mask a lost
    // target drop as success, and a tool breaking (-1) cancel a real +1 gain.
    // The drop set is the block's loot table union its own item name; the loot
    // lookup uses the CANONICAL block name (the env aliases LLM-favored names
    // like lapis_lazuli_ore onto real block objects, and wall_* variants have
    // no loot entries) so a legitimate mine is never filtered to zero.
    const canonicalName = blockByName.name;
    const dropNames = new Set([name, canonicalName]);
    const dealiasedName = canonicalName.replace(/(^|_)wall_/, "$1");
    dropNames.add(dealiasedName);
    const loot = mcData.blockLoot &&
        (mcData.blockLoot[canonicalName] || mcData.blockLoot[dealiasedName]);
    if (loot && Array.isArray(loot.drops)) {
        for (const d of loot.drops) {
            if (d && d.item) dropNames.add(d.item);
        }
    }
    const countItems = (dropsOnly) => {
        let total = 0;
        for (const item of bot.inventory.items()) {
            if (!dropsOnly || dropNames.has(item.name)) total += item.count;
        }
        return total;
    };
    const dropsBefore = countItems(true);
    const allBefore = countItems(false);

    await collectWithTimeout(bot, targets, {
        ignoreNoPath: true,
        count: count,
    }, count, name);

    // Wait for inventory sync to complete (so subsequent operations see the latest inventory state)
    await bot.waitForTicks(2);

    // P1: check whether the TARGET'S drops changed
    const dropsAfter = countItems(true);
    const allAfter = countItems(false);
    // clamp: the pathfinder can SPEND a same-name scaffolding item
    // (stone/cobblestone are scaffolding blocks), which would otherwise
    // produce a negative "items gained"
    const gained = Math.max(0, dropsAfter - dropsBefore);
    const junk = (allAfter - allBefore) - (dropsAfter - dropsBefore);
    if (gained === 0) {
        _mineBlockZeroGainCount++;
        // Diagnose the observed state instead of guessing, and surface it via
        // chat (console.warn never reaches the optimizer's feedback).
        const p = bot.entity ? bot.entity.position : null;
        const posStr = p
            ? `(${Math.floor(p.x)}, ${Math.floor(p.y)}, ${Math.floor(p.z)})`
            : "(unknown)";
        const blockAtBot = p ? bot.blockAt(p) : null;
        const held = bot.heldItem ? bot.heldItem.name : "an empty hand";
        let reason;
        if (blockAtBot && blockAtBot.name.includes("water")) {
            reason = `the bot is in water at ${posStr}; drops likely sank or drifted away`;
        } else if (blockAtBot && blockAtBot.name.includes("lava")) {
            reason = `the bot is in lava at ${posStr}; drops likely burned`;
        } else if (!targets[0].canHarvest(bot.heldItem ? bot.heldItem.type : null)) {
            reason = `the held tool changed or broke mid-mining (now holding ${held}), which cannot harvest ${name}`;
        } else {
            reason = `blocks were found but no drop entered the inventory (holding ${held} at ${posStr}); the drops or remaining blocks are likely unreachable`;
        }
        const junkNote = junk > 0
            ? ` (picked up ${junk} unrelated items en route, which do not count toward ${name})`
            : "";
        bot.chat(`Mined ${name} but gained none of its drops [${[...dropNames].join("/")}]: ${reason}${junkNote} (consecutive: ${_mineBlockZeroGainCount})`);
        if (_mineBlockZeroGainCount > 3) {
            _mineBlockZeroGainCount = 0;
            throw new Error(
                `mineBlock got no ${name} drops 3 consecutive times: ${reason}.`
            );
        }
    } else {
        _mineBlockZeroGainCount = 0;
    }

    bot.save(`${name}_mined`);
    // Return the target's drop items actually gained (may be less than count
    // if blocks were unreachable or drops were lost)
    return gained;
}
