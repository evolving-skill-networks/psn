// Use helper functions from adaptive_helpers.js (loaded into the global scope via control primitives)
// - pillarUp(bot, item)  — place a block at the bot's foot by jumping + placing below
// Vec3 / GoalPlaceBlock / gotoWithTimeout are provided by the /step scope; do not redeclare.

async function placeItem(bot, name, position) {
    // name must be a string
    if (typeof name !== "string") {
        throw new Error(`name for placeItem must be a string`);
    }
    // Defensively convert plain {x, y, z} objects to Vec3
    if (!(position instanceof Vec3)) {
        if (position && typeof position.x === 'number' && typeof position.y === 'number' && typeof position.z === 'number') {
            position = new Vec3(position.x, position.y, position.z);
        } else {
            throw new Error(`position for placeItem must be a Vec3 or {x, y, z}`);
        }
    }
    position = position.floored();
    const itemByName = mcData.itemsByName[name];
    if (!itemByName) {
        throw new Error(`No item named ${name}`);
    }
    const item = bot.inventory.findInventoryItem(itemByName.id);
    if (!item) {
        bot.chat(`No ${name} in inventory`);
        throw new Error(`No ${name} in inventory for placing`);
    }
    const item_count = item.count;

    // ===== Strategy A: the target is a cell the bot occupies (its foot or head) =====
    // Handing an occupied target to GoalPlaceBlock makes the pathfinder tower-up + dig
    // in a loop (it can never stand at a cell it must keep clear). Place AT the target by
    // pillaring up instead: jump to vacate the foot and place on the floor below. The head
    // cell reduces to the foot case after one filler pillar.
    const foot = bot.entity.position.floored();
    const head = foot.offset(0, 1, 0);
    const occupiesFoot = position.equals(foot);
    const occupiesHead = position.equals(head);
    // Foot needs no scaffolding (the item itself is the pillar block). Head needs a filler
    // to raise the bot one level first — only feasible if scaffolding is on hand. Without
    // scaffolding the pathfinder can't tower-up (so it won't thrash); fall through to
    // Strategy B, which places an occupied head by digging a reachable face.
    let usePillarUp = occupiesFoot;
    if (occupiesHead) {
        const mv = bot.pathfinder && bot.pathfinder.movements;
        const scaffold = (mv && typeof mv.getScaffoldingItem === "function") ? mv.getScaffoldingItem() : null;
        usePillarUp = !!scaffold;
    }
    if (usePillarUp) {
        bot.chat(`Target ${position} is the bot's ${occupiesFoot ? "foot" : "head"} cell; pillaring up to place ${name}`);
        // Raise the bot with filler blocks until its foot reaches the target level. An
        // occupied target is at most one above the foot, so this runs at most once; cap
        // iterations to guarantee termination (no runaway tower).
        let guard = 0;
        while (bot.entity.position.floored().y < position.y) {
            if (guard++ >= 2) {
                throw new Error(`Failed to place ${name}: could not pillar up to target ${position}`);
            }
            await pillarUp(bot);   // filler from getScaffoldingItem()
        }
        // The bot's foot is now at the target cell; pillar-up-place the item itself there.
        // A throw here still falls through to the blockAt/item-count reconciliation below
        // (symmetric with Strategy B's false-negative handling).
        try {
            await pillarUp(bot, item);
        } catch (e) {
            // verified by the world-state / item-count checks below
        }
        const placed = bot.blockAt(position);
        if (placedNameMatches(bot, name, placed)) {
            bot.chat(`Placed ${name} at (${position.x}, ${position.y}, ${position.z})`);
            return;
        }
        // false-negative tolerance: item consumed -> placement actually succeeded,
        // but the cell no longer holds it (gravity block fell, plant converted)
        const afterItem = bot.inventory.findInventoryItem(itemByName.id);
        if (!afterItem || afterItem.count < item_count) {
            bot.chat(`Placed ${name} at (${position.x}, ${position.y}, ${position.z}) but it fell or converted; the cell now holds ${placed ? placed.name : "nothing"}`);
            return;
        }
        throw new Error(`Failed to place ${name} at occupied target ${position}`);
    }

    // ===== Strategy B: place against a reference block =====
    // Reached for a non-occupied target, OR an occupied head with no scaffolding to pillar
    // with. In the latter case the pathfinder cannot tower-up, so GoalPlaceBlock will not
    // thrash — it digs a reachable face and places.
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
            bot.chat(`Placing ${name} on ${block.name} at ${block.position}`);
            break;
        }
    }
    if (!referenceBlock) {
        // No solid neighbour: the target is floating. Fail fast — do NOT relocate to a
        // different cell (relocation is the caller's job, not placeItem's).
        bot.chat(`No block to place ${name} on. You cannot place a floating block.`);
        throw new Error(
            `Failed to place ${name}: no valid reference block found. Cannot place a floating block.`
        );
    }

    try {
        // gotoWithTimeout prevents infinite blocking. The tower-up thrash only happens
        // when the bot both occupies the target AND has scaffolding — that case went to
        // Strategy A — so GoalPlaceBlock will not thrash here.
        await gotoWithTimeout(bot, new GoalPlaceBlock(position, bot.world, {}), 30000, `place position for ${name}`);
        await bot.equip(item, "hand");
        await bot.placeBlock(referenceBlock, faceVector);
        // Gravity blocks convert to a falling entity two server ticks after the
        // set; wait for the dust to settle so sand placed over a drop reports
        // honestly instead of claiming the cell still holds it.
        await bot.waitForTicks(4);
        const placedB = bot.blockAt(position);
        if (placedNameMatches(bot, name, placedB)) {
            bot.chat(`Placed ${name} at (${position.x}, ${position.y}, ${position.z})`);
        } else {
            bot.chat(`Placed ${name} at (${position.x}, ${position.y}, ${position.z}) but it fell or converted; the cell now holds ${placedB ? placedB.name : "nothing"}`);
        }
    } catch (err) {
        const afterItem = bot.inventory.findInventoryItem(itemByName.id);
        if (afterItem?.count === item_count) {
            // item still in inventory -> placement really failed. The server
            // silently rejects placing a block into a cell an entity stands in
            // (no blockUpdate fires), so the raw error reads as an empty-air
            // timeout. Surface the blocking entity so the caller can perceive
            // and decide how to handle it (kill it, lure it away, place elsewhere).
            const blocker = entityOccupyingCell(bot, position);
            const blockedNote = blocker
                ? ` A ${entityLabel(blocker)} is occupying the target cell ${position}; a block cannot be placed where an entity stands.`
                : "";
            bot.chat(`Error placing ${name}: ${err.message}, please find another position to place`);
            throw new Error(`Failed to place ${name}: ${err.message}.${blockedNote}`);
        } else {
            // item count decreased -> placement actually succeeded (mineflayer API false negative)
            bot.chat(`Placed ${name} at (${position.x}, ${position.y}, ${position.z})`);
        }
    }
}

// The post-placement read must accept every block this item can legitimately
// become (wall variants like torch -> wall_torch, bucket fluids); falls back
// to exact-name matching when the place_block patch is absent.
function placedNameMatches(bot, itemName, blk) {
    if (!blk) return false;
    const expected = bot._expectedPlacedBlockNames ? bot._expectedPlacedBlockNames(itemName) : null;
    return expected ? expected.has(blk.name) : blk.name === itemName;
}

// A block cannot be placed where an entity stands. Return the entity whose body
// overlaps `cell` (a floored Vec3), or null. Mob/player bodies are ~0.6 wide and
// up to ~2 tall, so test the entity's horizontal centre against the cell expanded
// by a half-width and its vertical span against the cell's [y, y+1). Dropped
// items, xp orbs and projectiles do not block placement and are ignored.
function entityOccupyingCell(bot, cell) {
    const HALF_W = 0.4;
    const SKIP = new Set(["object", "orb", "projectile", "global", "other"]);
    let best = null;
    let bestDist = Infinity;
    for (const id in bot.entities) {
        const e = bot.entities[id];
        if (!e || e === bot.entity || !e.position) continue;
        if (e.type && SKIP.has(e.type)) continue;
        if (e.name === "item" || e.name === "experience_orb" || e.name === "arrow") continue;
        const ep = e.position;
        const height = e.height && e.height > 0 ? e.height : 1.8;
        const horizIn =
            ep.x >= cell.x - HALF_W && ep.x <= cell.x + 1 + HALF_W &&
            ep.z >= cell.z - HALF_W && ep.z <= cell.z + 1 + HALF_W;
        const vertIn = ep.y <= cell.y + 1 && ep.y + height >= cell.y;
        if (horizIn && vertIn) {
            const d = ep.distanceTo(cell.offset(0.5, 0.5, 0.5));
            if (d < bestDist) {
                bestDist = d;
                best = e;
            }
        }
    }
    return best;
}

function entityLabel(e) {
    return e.username || e.name || e.displayName || e.kind || "entity";
}
