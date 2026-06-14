// Mechanism context for kill-failure messages: where the target is NOW
// relative to the bot, plus the bot's health. Knockback and fleeing are the
// common silent causes; surfacing them lets the optimizer diagnose the real
// failure instead of guessing from a bare "Failed to kill".
function _mobFailContext(bot, entity) {
    try {
        const d = entity && entity.position && bot.entity
            ? entity.position.distanceTo(bot.entity.position).toFixed(1)
            : "?";
        const pos = entity && entity.position
            ? `(${Math.floor(entity.position.x)}, ${Math.floor(entity.position.y)}, ${Math.floor(entity.position.z)})`
            : "(unknown)";
        const hp = bot.health !== undefined ? `${Math.round(bot.health)}/20` : "?";
        return `target still alive at ${pos}, ${d} blocks away; bot health ${hp}`;
    } catch (e) {
        return "no further state available";
    }
}

function waitForMobRemoved(bot, entity, timeout = 300) {
    return new Promise((resolve, reject) => {
        let success = false;
        let droppedItem = null;
        let done = false;

        function cleanUp() {
            if (done) return;
            done = true;
            clearTimeout(timeoutId);
            bot.removeListener("entityGone", onEntityGone);
            bot.removeListener("stoppedAttacking", onStoppedAttacking);
            bot.removeListener("itemDrop", onItemDrop);
        }

        // Timeout rejects directly — prevents promise hang if stoppedAttacking never fires
        const timeoutId = setTimeout(() => {
            bot.pvp.stop();
            cleanUp();
            reject(new Error(`Failed to kill ${entity.name} within ${timeout}s timeout: ${_mobFailContext(bot, entity)}. The mob likely fled or the bot was knocked out of range.`));
        }, timeout * 1000);

        function onEntityGone(e) {
            if (e === entity) {
                success = true;
                bot.chat(`Killed ${entity.name}!`);
                bot.pvp.stop();
            }
        }

        function onItemDrop(item) {
            if (entity.position.distanceTo(item.position) <= 1) {
                droppedItem = item;
            }
        }

        function onStoppedAttacking() {
            cleanUp();
            if (success) resolve(droppedItem);
            else reject(new Error(`Failed to kill ${entity.name}: the attack stopped before the kill (${_mobFailContext(bot, entity)}).`));
        }

        bot.on("entityGone", onEntityGone);
        bot.on("stoppedAttacking", onStoppedAttacking);
        bot.on("itemDrop", onItemDrop);
    });
}


function waitForMobShot(bot, entity, timeout = 300) {
    return new Promise((resolve, reject) => {
        let success = false;
        let droppedItem = null;
        let done = false;

        function cleanUp() {
            if (done) return;
            done = true;
            clearTimeout(timeoutId);
            bot.removeListener("entityGone", onEntityGone);
            bot.removeListener("auto_shot_stopped", onAutoShotStopped);
            bot.removeListener("itemDrop", onItemDrop);
        }

        // Timeout rejects directly — prevents promise hang if auto_shot_stopped never fires
        const timeoutId = setTimeout(() => {
            bot.hawkEye.stop();
            cleanUp();
            reject(new Error(`Failed to shoot ${entity.name} within ${timeout}s timeout: ${_mobFailContext(bot, entity)}.`));
        }, timeout * 1000);

        function onEntityGone(e) {
            if (e === entity) {
                success = true;
                bot.chat(`Shot ${entity.name}!`);
                bot.hawkEye.stop();
            }
        }

        function onItemDrop(item) {
            if (entity.position.distanceTo(item.position) <= 1) {
                droppedItem = item;
            }
        }

        function onAutoShotStopped() {
            cleanUp();
            if (success) resolve(droppedItem);
            else reject(new Error(`Failed to shoot ${entity.name}: shooting stopped before the kill (${_mobFailContext(bot, entity)}).`));
        }

        bot.on("entityGone", onEntityGone);
        bot.on("auto_shot_stopped", onAutoShotStopped);
        bot.on("itemDrop", onItemDrop);
    });
}
