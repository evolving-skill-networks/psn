async function smeltItem(bot, itemName, fuelName, count = 1) {
    // return if itemName or fuelName is not string
    if (typeof itemName !== "string" || typeof fuelName !== "string") {
        throw new Error("itemName or fuelName for smeltItem must be a string");
    }
    // return if count is not a number
    if (typeof count !== "number") {
        throw new Error("count for smeltItem must be a number");
    }
    const item = mcData.itemsByName[itemName];
    const fuel = mcData.itemsByName[fuelName];
    if (!item) {
        throw new Error(`No item named ${itemName} (the item to smelt) exists`);
    }
    if (!fuel) {
        throw new Error(`No item named ${fuelName} (the fuel) exists`);
    }
    const furnaceBlock = bot.findBlock({
        matching: mcData.blocksByName.furnace.id,
        maxDistance: 32,
    });
    if (!furnaceBlock) {
        throw new Error("No furnace nearby");
    } else {
        await gotoWithTimeout(bot, new GoalLookAtBlock(furnaceBlock.position, bot.world), 30000, 'furnace');
    }
    let furnace = await bot.openFurnace(furnaceBlock);
    let success_count = 0;
    // Reason for an early break (no input / no fuel) so the FINAL thrown error
    // carries the specific cause instead of a generic "check fuel and input".
    let failReason = null;
    const countItem = (id) =>
        bot.inventory.items().filter((x) => x.type === id).reduce((a, x) => a + x.count, 0);
    const nearbyHostile = () => bot.entity ? bot.nearestEntity(
        (e) => e && e.position && (e.type === "mob" || e.type === "hostile") &&
               e.position.distanceTo(bot.entity.position) < 8) : null;

    for (let i = 0; i < count; i++) {
        if (!bot.inventory.findInventoryItem(item.id, null)) {
            failReason = `no ${itemName} left in inventory to smelt`;
            break;
        }
        if (furnace.fuelSeconds < 15 && furnace.fuelItem()?.name !== fuelName) {
            if (!bot.inventory.findInventoryItem(fuel.id, null)) {
                failReason = `no ${fuelName} in inventory to use as fuel`;
                break;
            }
            await furnace.putFuel(fuel.id, null, 1);
            await bot.waitForTicks(20);
            if (!furnace.fuel && furnace.fuelItem()?.name !== fuelName) {
                throw new Error(`${fuelName} is not a valid fuel`);
            }
        }
        const inputBefore = countItem(item.id);
        await furnace.putInput(item.id, null, 1);
        await bot.waitForTicks(12 * 20);
        if (!furnace.outputItem()) {
            // No output after the smelt time. DON'T blame the input by default:
            // diagnose the real cause from mechanistic state.
            const hostile = nearbyHostile();
            const hostileNote = hostile
                ? ` (a ${hostile.name || hostile.displayName || "mob"} is nearby and may have knocked the bot away)` : "";
            const dist = (bot.entity && furnaceBlock)
                ? bot.entity.position.distanceTo(furnaceBlock.position) : null;

            // (a) Still out of interaction range -> the smelt was interrupted.
            if (dist != null && dist > 6) {
                throw new Error(
                    `Failed to smelt ${itemName}: the bot is ${dist.toFixed(1)} blocks from the ` +
                    `furnace (interaction needs <= ~6), so the smelt was interrupted${hostileNote}. ` +
                    `The input is valid; re-approach the furnace and retry.`);
            }

            // In range now, but the furnace view may be STALE if the bot was
            // briefly knocked out of range during the smelt (the server may have
            // actually produced output). Re-open to refresh and re-check before
            // concluding anything.
            let refreshed = null;
            try { furnace.close(); } catch (e) { /* ignore */ }
            try { refreshed = await bot.openFurnace(furnaceBlock); } catch (e) { refreshed = null; }
            if (refreshed) {
                furnace = refreshed;
                if (furnace.outputItem()) {
                    // Stale false-negative: it actually smelted. Recover.
                    await furnace.takeOutput();
                    success_count++;
                    continue;
                }
            }

            // Still genuinely no output. Use signals to name the real cause.
            const inputConsumed = countItem(item.id) < inputBefore;   // did putInput take effect?
            const inputStuck = !!(refreshed && furnace.inputItem() && furnace.inputItem().type === item.id);
            const hasFuel = !!(refreshed && (((furnace.fuel || 0) > 0) || furnace.fuelItem()));

            if (!refreshed) {
                // Could not re-open: the furnace is no longer reachable (destroyed
                // or the bot was pushed away) -> interrupted, not invalid input.
                throw new Error(
                    `Failed to smelt ${itemName}: lost access to the furnace (it may have been ` +
                    `destroyed or the bot pushed out of range)${hostileNote}, so the smelt was ` +
                    `interrupted. Re-approach a furnace and retry.`);
            }
            if (!inputConsumed && !inputStuck) {
                // The item never made it into the furnace (interaction failed even
                // though the bot is back in range, e.g. knocked away mid-putInput).
                throw new Error(
                    `Failed to smelt ${itemName}: the input was not placed into the furnace ` +
                    `(interaction failed${hostileNote}), so nothing smelted. The input is valid; ` +
                    `re-approach the furnace and retry.`);
            }
            if (!hasFuel) {
                // Input went in but the furnace has no fuel -> ran out mid-smelt.
                throw new Error(
                    `Failed to smelt ${itemName}: the furnace ran out of fuel before smelting ` +
                    `finished. Provide more ${fuelName} and retry.`);
            }
            // Input went in, fuel is present, yet nothing was produced after the
            // smelt time -> the item genuinely is not a smeltable input.
            throw new Error(
                `${itemName} is not a valid input (it sat in the furnace with fuel present ` +
                `but did not smelt into anything).`);
        }
        await furnace.takeOutput();
        success_count++;
    }
    try { furnace.close(); } catch (e) { /* ignore close errors — furnace may be destroyed */ }
    if (success_count > 0) {
        bot.chat(`Smelted ${success_count} ${itemName}.`);
        return;
    }
    // Nothing smelted: surface the specific cause (no input / no fuel) in the
    // THROWN error, not just a chat, so the caller and optimizer receive it.
    const msg = failReason
        ? `Failed to smelt ${itemName}: ${failReason}.`
        : `Failed to smelt ${itemName}: produced no output (check the fuel and input).`;
    bot.chat(msg);
    _smeltItemFailCount++;
    throw new Error(msg);
}
