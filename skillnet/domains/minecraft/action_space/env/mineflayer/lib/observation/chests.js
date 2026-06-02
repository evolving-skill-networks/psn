const { Observation } = require("./base");

class Chests extends Observation {
    constructor(bot) {
        super(bot);
        this.name = "nearbyChests";
        this.chestsItems = {};
        bot.on("closeChest", (chestItems, position) => {
            this.chestsItems[position] = chestItems;
        });
        bot.on("removeChest", (chestPosition) => {
            this.chestsItems[chestPosition] = "Invalid";
        });
    }

    observe() {
        try {
            // Defensive check: ensure registry and blocksByName exist
            if (!this.bot.registry || !this.bot.registry.blocksByName || !this.bot.registry.blocksByName.chest) {
                return {};
            }
            
        const chests = this.bot.findBlocks({
            matching: this.bot.registry.blocksByName.chest.id,
            maxDistance: 16,
            count: 999,
        });
            
            if (!chests || !Array.isArray(chests)) {
                return this.chestsItems;
            }
            
        chests.forEach((chest) => {
                // Convert Vec3 to string key
                const key = chest ? chest.toString() : String(chest);
                if (!this.chestsItems.hasOwnProperty(key)) {
                    this.chestsItems[key] = "Unknown";
            }
        });
        return this.chestsItems;
        } catch (err) {
            console.error(`[Chests] observe error: ${err.message}`);
            return {};
        }
    }
}

module.exports = Chests;
