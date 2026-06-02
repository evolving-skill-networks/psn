function failedCraftFeedback(bot, name, item, craftingTable) {
    const recipes = bot.recipesAll(item.id, null, craftingTable);
    if (!recipes.length) {
        // Distinguish between different failure reasons
        if (craftingTable === null) {
            // Try to look up whether a recipe exists that requires a crafting table
            const tableRecipes = bot.recipesAll(item.id, null, true);
            if (tableRecipes.length) {
                throw new Error(`${name} requires a crafting table (3x3 grid). Place or find a crafting table first.`);
            } else {
                throw new Error(`No recipe found for ${name}. Check if the item name is correct.`);
            }
        } else {
            throw new Error(`No crafting table nearby or no valid recipe for ${name}`);
        }
    } else {
        const recipes = bot.recipesAll(
            item.id,
            null,
            mcData.blocksByName.crafting_table.id
        );
        // Calculate the missing material count for each recipe
        var min = 999;
        const recipeMissingMap = [];
        for (const recipe of recipes) {
            const delta = recipe.delta;
            var missing = 0;
            for (const delta_item of delta) {
                if (delta_item.count < 0) {
                    const inventory_item = bot.inventory.findInventoryItem(
                        mcData.items[delta_item.id].name,
                        null
                    );
                    if (!inventory_item) {
                        missing += -delta_item.count;
                    } else {
                        missing += Math.max(
                            -delta_item.count - inventory_item.count,
                            0
                        );
                    }
                }
            }
            recipeMissingMap.push({ recipe, missing });
            if (missing < min) {
                min = missing;
            }
        }
        // Collect all recipes with the fewest missing materials (tied recipes)
        const tiedRecipes = recipeMissingMap
            .filter(r => r.missing === min)
            .map(r => r.recipe);
        const min_recipe = tiedRecipes[0];
        const delta = min_recipe.delta;
        let message = "";
        // Collect the list of missing materials and detect variant groups
        const missingItems = [];
        for (const delta_item of delta) {
            if (delta_item.count < 0) {
                const itemName = mcData.items[delta_item.id].name;
                const inventory_item = bot.inventory.findInventoryItem(
                    itemName,
                    null
                );
                const required = -delta_item.count;
                const have = inventory_item ? inventory_item.count : 0;
                if (have < required) {
                    const need = required - have;

                    // Detect variant groups: when multiple tied recipes share the same count at this position but different items
                    if (tiedRecipes.length > 1) {
                        const variantNames = _collectCraftVariantNames(tiedRecipes, delta_item);
                        if (variantNames.length > 1) {
                            const variantMsg = `${need} planks (any single type: ${variantNames.join(', ')})`;
                            missingItems.push(variantMsg);
                            message += `${need} more planks (any type), `;
                            continue;
                        }
                    }

                    missingItems.push(`${need} ${itemName}`);
                    message += `${need} more ${itemName}, `;
                }
            }
        }
        bot.chat(`I cannot make ${name} because I need: ${message}`);
        // Throw an error containing detailed material info so the caller knows the reason for failure
        throw new Error(`Insufficient materials to craft ${name}. Missing: ${missingItems.join(', ')}`);
    }
}

/**
 * Collect variant names at the same position across tied recipes
 * Used to detect recipe variant groups (e.g. oak_planks/birch_planks/spruce_planks)
 */
function _collectCraftVariantNames(tiedRecipes, referenceDelta) {
    const variants = new Set();
    for (const recipe of tiedRecipes) {
        for (const delta of recipe.delta) {
            if (delta.count === referenceDelta.count && delta.id !== referenceDelta.id) {
                variants.add(mcData.items[delta.id].name);
            }
        }
    }
    // Also include the reference itself
    if (variants.size > 0) {
        variants.add(mcData.items[referenceDelta.id].name);
    }
    return Array.from(variants).sort();
}
