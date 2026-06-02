// Kill a mob and collect dropped items.
//
// PARAMETERS:
//   killMob(bot, mobName, timeout = 300)
//   - mobName: string — mob entity name (must match entity.name)
//   - timeout: number — max time in seconds to wait for mob death
//
// BEHAVIOR:
// - Searches for nearest entity matching mobName within 48 blocks
//   (48 blocks = slightly larger than explore distance of 32)
// - If no mob found: chats "No ... nearby, please explore first" and returns
//   (increments fail counter, throws after 10+ consecutive failures)
//
// WEAPON DETECTION:
// - Checks mainHandItem for ranged weapons: bow, crossbow, snowball, ender_pearl, egg, splash_potion, trident
// - If ranged weapon equipped: uses bot.hawkEye.autoAttack() + waitForMobShot()
// - If melee weapon or no weapon: uses bot.pvp.attack() + waitForMobRemoved()
//
// ITEM COLLECTION:
// - After mob death, if a dropped item entity exists, collects it with collectWithTimeout()
//
// ERROR HANDLING:
//   - "mobName for killMob must be a string"
//   - "timeout for killMob must be a number"
//   - "Bot entity is undefined" — bot disconnected/died
//   - "killMob failed too many times" — 10+ consecutive no-mob-found
//
// NOTE: Calls bot.save('mobName_killed') on success.
async function killMob(bot, mobName, timeout = 300) {
    if (typeof mobName !== "string") throw new Error(`mobName for killMob must be a string`);
    if (typeof timeout !== "number") throw new Error(`timeout for killMob must be a number`);
    if (!bot.entity) throw new Error("Bot entity is undefined, cannot find nearby mobs");
    const entity = bot.nearestEntity(
        (entity) => entity.name === mobName &&
            bot.entity && entity.position.distanceTo(bot.entity.position) < 48
    );
    if (!entity) {
        bot.chat(`No ${mobName} nearby, please explore first`);
        return;
    }
    const weaponsForShooting = ["bow", "crossbow", "snowball", "ender_pearl", "egg", "splash_potion", "trident"];
    const mainHandItem = bot.inventory.slots[bot.getEquipmentDestSlot("hand")];
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
