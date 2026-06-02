const { Observation } = require("./base");

class Inventory extends Observation {
    constructor(bot) {
        super(bot);
        this.name = "inventory";
    }

    observe() {
        try {
            return listItems(this.bot);
        } catch (err) {
            console.error(`[Inventory Observation] Error in observe(): ${err.message}`);
            if (err.stack) {
                console.error(`[Inventory Observation] Stack: ${err.stack}`);
            }
            // Return empty object instead of throwing, to ensure observe flow can continue
            return {};
        }
    }
}

function listItems(bot) {
    const items = getInventoryItems(bot);
    return items.reduce(itemToDict, {});
}

function getInventoryItems(bot) {
    try {
        const inventory = bot.currentWindow || bot.inventory;
        if (!inventory) {
            console.warn("[Inventory Observation] bot.currentWindow and bot.inventory are both undefined");
            return [];
        }
        if (typeof inventory.items !== 'function') {
            console.warn(`[Inventory Observation] inventory.items is not a function, type: ${typeof inventory.items}`);
            return [];
        }
        return inventory.items();
    } catch (err) {
        console.error(`[Inventory Observation] Error getting inventory items: ${err.message}`);
        if (err.stack) {
            console.error(`[Inventory Observation] Stack: ${err.stack}`);
        }
        return [];
    }
}

function itemToDict(acc, cur) {
    if (cur.name && cur.count) {
        //if both name and count property are defined
        if (acc[cur.name]) {
            //if the item is already in the dict
            acc[cur.name] += cur.count;
        } else {
            //if the item is not in the dict
            acc[cur.name] = cur.count;
        }
    }
    return acc;
}

//export modules
module.exports = Inventory;
