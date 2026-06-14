async function killMob(bot, mobName, timeout = 300) {
    // return if mobName is not string
    if (typeof mobName !== "string") {
        throw new Error(`mobName for killMob must be a string`);
    }
    // return if timeout is not number
    if (typeof timeout !== "number") {
        throw new Error(`timeout for killMob must be a number`);
    }

    const weaponsForShooting = [
        "bow",
        "crossbow",
        "snowball",
        "ender_pearl",
        "egg",
        "splash_potion",
        "trident",
    ];
    const mainHandItem = bot.inventory.slots[bot.getEquipmentDestSlot("hand")];

    // P0: bot.entity may be undefined on death/disconnect
    if (!bot.entity) {
        throw new Error("Bot entity is undefined, cannot find nearby mobs");
    }
    const entity = bot.nearestEntity(
        (entity) =>
            entity.name === mobName &&
            bot.entity &&  // guard against mid-search disconnect
            // kill mob distance should be slightly bigger than explore distance
            entity.position.distanceTo(bot.entity.position) < 48
    );
    if (!entity) {
        // Be specific about what was searched: the radius, the bot position,
        // and whether the mob exists just beyond range (approach vs explore).
        const p = bot.entity.position;
        const here = `within 48 blocks of (${Math.floor(p.x)}, ${Math.floor(p.y)}, ${Math.floor(p.z)})`;
        const beyond = bot.nearestEntity(
            (e) => e.name === mobName && e.position && bot.entity
        );
        const hint = beyond && beyond.position
            ? `; the nearest ${mobName} is ${beyond.position.distanceTo(p).toFixed(0)} blocks away, approach it first`
            : `, please explore first`;
        bot.chat(`No ${mobName} ${here}${hint}`);
        _killMobFailCount++;
        if (_killMobFailCount > 10) {
            throw new Error(
                `killMob failed too many times: no ${mobName} ${here}${hint}.`
            );
        }
        return;
    }

    let droppedItem;
    if (mainHandItem && weaponsForShooting.includes(mainHandItem.name)) {
        bot.hawkEye.autoAttack(entity, mainHandItem.name);
        droppedItem = await waitForMobShot(bot, entity, timeout);
    } else {
        await bot.pvp.attack(entity);
        droppedItem = await waitForMobRemoved(bot, entity, timeout);
    }
    if (droppedItem) {
        await collectWithTimeout(bot, droppedItem, { ignoreNoPath: true }, 1, 'dropped items');
    }
    bot.save(`${mobName}_killed`);
}
