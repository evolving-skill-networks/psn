const Observation = require("./base.js").Observation;

// Structured trace of what the bot DID during a step: block placements and
// digs (with coordinates), item pickups, combat events, knockback, and
// damage taken. Skill chat narration only reports what its author chose to
// say; this records the ground truth from mineflayer's bot-caused events so
// the optimizer can diagnose action-level causes (e.g. a scaffold block
// placed into a cell that must stay clear).
//
// NOTE: the name deliberately has no "on" prefix. "on*" observers trigger a
// full observation snapshot per event (see base.js); this buffer instead
// rides the snapshots that are already being taken (chat messages, errors,
// the final observe), adding no extra snapshot cost.
class BlockActions extends Observation {
    constructor(bot) {
        super(bot);
        this.name = "blockActions";
        this.obs = [];
        this.maxLen = 100;
        this.dropped = 0;

        // Ring semantics: when a window overflows, evict the OLDEST entry so
        // the buffer keeps the tail, consistent with the Python step-cap and
        // the prompt render, which also keep the most recent actions (the
        // tail is what matters when diagnosing the failure at step end).
        const push = (entry) => {
            if (this.obs.length >= this.maxLen) {
                this.obs.shift();
                this.dropped++;
            }
            this.obs.push(entry);
        };
        const near = (pos, r) => {
            try {
                return !!(bot.entity && pos && pos.distanceTo(bot.entity.position) <= r);
            } catch (e) {
                return false;
            }
        };

        // Bot-caused only: emitted from bot.placeBlock (place_block.js).
        bot.on("blockPlaced", (oldBlock, newBlock) => {
            if (!newBlock || !newBlock.position) return;
            const p = newBlock.position;
            push({ t: "place", name: newBlock.name, x: p.x, y: p.y, z: p.z });
        });
        // Bot-caused only: emitted from bot.dig. The event argument is the
        // POST-dig block (air), so only the position is trustworthy here.
        bot.on("diggingCompleted", (block) => {
            if (!block || !block.position) return;
            const p = block.position;
            push({ t: "dig", x: p.x, y: p.y, z: p.z });
        });
        bot.on("diggingAborted", (block) => {
            if (!block || !block.position) return;
            const p = block.position;
            push({ t: "dig_aborted", name: block.name, x: p.x, y: p.y, z: p.z });
        });
        // Item entered the bot's inventory from the ground.
        bot.on("playerCollect", (collector, collected) => {
            if (!bot.entity || collector !== bot.entity) return;
            let name = "item";
            let count = 1;
            try {
                const it = collected.getDroppedItem && collected.getDroppedItem();
                if (it) {
                    name = it.name;
                    count = it.count;
                }
            } catch (e) {
                // keep defaults; the pickup itself is still worth recording
            }
            push({ t: "collect", name: name, count: count });
        });
        // Combat signal: damage to the bot itself, or to a mob close enough
        // to plausibly be the bot's target.
        bot.on("entityHurt", (entity) => {
            if (!entity) return;
            if (bot.entity && entity === bot.entity) {
                push({ t: "bot_hurt", hp: Math.round(bot.health || 0) });
            } else if (entity.position && near(entity.position, 16)) {
                push({
                    t: "hit",
                    name: entity.name,
                    d: +entity.position.distanceTo(bot.entity.position).toFixed(1),
                });
            }
        });
        bot.on("entityDead", (entity) => {
            if (entity && entity.position && near(entity.position, 32)) {
                push({ t: "killed", name: entity.name });
            }
        });
        // Server moved the bot (knockback, teleport): the classic silent
        // cause behind "out of interaction range" failures.
        bot.on("forcedMove", () => {
            if (!bot.entity) return;
            const p = bot.entity.position;
            push({
                t: "forced_move",
                x: Math.floor(p.x),
                y: Math.floor(p.y),
                z: Math.floor(p.z),
            });
        });
    }

    observe() {
        const result = this.obs;
        if (this.dropped > 0) {
            // The evicted entries are the oldest, so the marker belongs at
            // the head: "N earlier actions were dropped".
            result.unshift({ t: "dropped", count: this.dropped });
        }
        this.obs = [];
        this.dropped = 0;
        return result;
    }
}

module.exports = BlockActions;
