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
            reject(new Error(`Failed to kill ${entity.name} within ${timeout}s timeout.`));
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
            else reject(new Error(`Failed to kill ${entity.name}.`));
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
            reject(new Error(`Failed to shoot ${entity.name} within ${timeout}s timeout.`));
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
            else reject(new Error(`Failed to shoot ${entity.name}.`));
        }

        bot.on("entityGone", onEntityGone);
        bot.on("auto_shot_stopped", onAutoShotStopped);
        bot.on("itemDrop", onItemDrop);
    });
}
