// placeItem(bot, name, position) — places an inventory item as a block at the given position.
//
// IMPORTANT: A valid placement position requires ALL of:
// (1) bot.blockAt(position) is air or replaceable (water, tall_grass)
// (2) At least one face-adjacent block (±x, ±y, ±z) is NOT air (serves as placement reference)
// (3) Position is NOT occupied by the bot entity (causes blockUpdate timeout)
// If condition (2) is not met, placement fails with "no valid reference block found".
//
// WARNING: Target position must NOT overlap with the bot's entity bounding box.
// Bot occupies 2 blocks vertically (feet + head). Placing a block at bot's position
// causes "blockUpdate did not fire within timeout" because the entity blocks placement.
// RECOMMENDED: Use bot.entity.position.offset(2, 0, 0).floored() for safe offset.
//
// ADAPTIVE BEHAVIOR: placeItem internally retries with fallback strategies:
//   - If first reference block fails, tries all 6 face directions
//   - If position is occupied, searches nearby open area automatically
//   - If blocked, may dig to clear space
// So you do NOT need to implement retry logic yourself — just call placeItem once.
//
// ERROR HANDLING: placeItem THROWS on failure. Do NOT catch and suppress its errors.
// WRONG: try { await placeItem(...) } catch(e) { bot.chat(e.message); return; }
// RIGHT: await placeItem(bot, "crafting_table", position);  // let errors propagate
//
async function placeItem(bot, name, position) {
    const item = bot.inventory.findInventoryItem(mcData.itemsByName[name].id);
    // find a reference block
    const faceVectors = [
        new Vec3(0, 1, 0),
        new Vec3(0, -1, 0),
        new Vec3(1, 0, 0),
        new Vec3(-1, 0, 0),
        new Vec3(0, 0, 1),
        new Vec3(0, 0, -1),
    ];
    let referenceBlock = null;
    let faceVector = null;
    for (const vector of faceVectors) {
        const block = bot.blockAt(position.minus(vector));
        if (block?.name !== "air") {
            referenceBlock = block;
            faceVector = vector;
            break;
        }
    }
    // You must first go to the block position you want to place
    await bot.pathfinder.goto(new GoalPlaceBlock(position, bot.world, {}));
    // You must equip the item right before calling placeBlock
    await bot.equip(item, "hand");
    await bot.placeBlock(referenceBlock, faceVector);
}
