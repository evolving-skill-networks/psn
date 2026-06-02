async function craftItem(bot, name, count = 1) {
    // return if name is not string
    if (typeof name !== "string") {
        throw new Error("name for craftItem must be a string");
    }
    // return if count is not number
    if (typeof count !== "number") {
        throw new Error("count for craftItem must be a number");
    }
    const itemByName = mcData.itemsByName[name];
    if (!itemByName) {
        throw new Error(`No item named ${name}`);
    }

    // Wait for inventory to sync (fixes the case where calling craftItem right after mineBlock sees stale inventory)
    await bot.waitForTicks(2);

    const craftingTable = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 32,
    });
    if (!craftingTable) {
        bot.chat("Craft without a crafting table");
    } else {
        await gotoWithTimeout(bot, new GoalLookAtBlock(craftingTable.position, bot.world), 30000, 'crafting table');
    }

    // Wait again to ensure inventory is fully in sync
    await bot.waitForTicks(1);

    // Use the enhanced checkRecipe API to verify that materials are sufficient for the full count
    const checkResult = bot.checkRecipe(name, count, craftingTable);

    if (checkResult.available) {
        bot.chat(`I can make ${name}`);

        // Craft one at a time to avoid mineflayer inventory sync issues
        // Batched craft may fail because inventory state is out of sync
        let successCount = 0;
        for (let i = 0; i < count; i++) {
            // Recheck recipe availability before each craft
            const singleCheck = bot.checkRecipe(name, 1, craftingTable);
            if (!singleCheck.available) {
                if (i === 0) {
                    bot.chat(singleCheck.message);
                    if (singleCheck.reason === 'insufficient_materials') {
                        failedCraftFeedback(bot, name, itemByName, craftingTable);
                    }
                    _craftItemFailCount++;
                    if (_craftItemFailCount > 10) {
                        throw new Error("craftItem failed too many times, check chat log to see what happened");
                    }
                    throw new Error(`craftItem failed for ${name}: ${singleCheck.message}`);
                }
                // Partial success
                bot.chat(`Crafted ${i}/${count} ${name}(s) before running out of materials`);
                break;
            }

            try {
                await bot.craft(singleCheck.recipe, 1, craftingTable);
                successCount++;
                await bot.waitForTicks(2);  // wait for inventory to sync
            } catch (err) {
                if (successCount === 0) {
                    bot.chat(`I cannot do the recipe for ${name}: ${err.message}`);
                    throw err;
                }
                bot.chat(`Crafted ${successCount}/${count} ${name}(s), stopped due to: ${err.message}`);
                break;
            }
        }

        if (successCount > 0) {
            bot.chat(`I did the recipe for ${name} ${successCount} times`);
        }
    } else {
        // Use the detailed error information provided by checkRecipe
        bot.chat(checkResult.message);

        // For insufficient-materials errors, still call failedCraftFeedback to provide more detailed feedback
        if (checkResult.reason === 'insufficient_materials') {
            failedCraftFeedback(bot, name, itemByName, craftingTable);
        }

        _craftItemFailCount++;
        if (_craftItemFailCount > 10) {
            throw new Error(
                "craftItem failed too many times, check chat log to see what happened"
            );
        }
        throw new Error(`craftItem failed for ${name}: ${checkResult.message}`);
    }
}
