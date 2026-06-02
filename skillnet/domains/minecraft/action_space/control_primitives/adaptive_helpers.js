/**
 * Environment-adaptation helpers
 * Used to support adaptive placement, pathfinding, mining, and similar operations
 *
 * These functions address environment-constraint issues such as:
 * - Placement issues: no space around the bot to place a block
 * - Pathfinding issues: the path is blocked and the target is unreachable
 * - Mining issues: the target block cannot be found
 * - Combat issues: entity position problems
 *
 * Note: this file is concatenated into the eval() context for execution.
 * Vec3 is already injected into the global scope via globalDepsCode — do not redeclare it!
 */

// GoalNear/GoalBlock are already declared with const in the /step route scope (index.js:441-455);
// they are accessible from eval via the scope chain — do not redeclare them!
// Vec3 is already injected by index.js's globalDepsCode — no declaration needed, use the global Vec3 directly

/**
 * Find a nearby open area
 * Used to find a suitable position when placing a block
 *
 * @param {Bot} bot - Mineflayer bot instance
 * @param {Vec3} center - search center position (optional; defaults to bot position)
 * @param {number} maxRadius - maximum search radius
 * @returns {Vec3|null} - the open position found, or null
 */
async function findNearbyOpenArea(bot, center, maxRadius = 8) {
    // P0: defensive check
    if (!center && !bot.entity) {
        console.warn("[findNearbyOpenArea] bot.entity is undefined and no center provided");
        return null;
    }
    const foot = center || bot.entity.position.floored();

    for (let r = 2; r <= maxRadius; r++) {
        for (let dx = -r; dx <= r; dx++) {
            for (let dz = -r; dz <= r; dz++) {
                // Only check edge positions (optimizes the search)
                if (Math.abs(dx) !== r && Math.abs(dz) !== r) continue;

                const checkPos = foot.offset(dx, 0, dz);
                const block = bot.blockAt(checkPos);
                const above = bot.blockAt(checkPos.offset(0, 1, 0));
                const below = bot.blockAt(checkPos.offset(0, -1, 0));

                // Check: current position is air, above is air, below is a solid block
                if (block && block.name === 'air' &&
                    above && above.name === 'air' &&
                    below && below.boundingBox === 'block') {

                    // Extra check: at least one neighbor is air (makes placement easier)
                    const neighbors = [
                        checkPos.offset(1, 0, 0),
                        checkPos.offset(-1, 0, 0),
                        checkPos.offset(0, 0, 1),
                        checkPos.offset(0, 0, -1)
                    ];
                    
                    let hasAirNeighbor = false;
                    for (const nPos of neighbors) {
                        const nBlock = bot.blockAt(nPos);
                        if (nBlock && nBlock.name === 'air') {
                            hasAirNeighbor = true;
                            break;
                        }
                    }
                    
                    if (hasAirNeighbor) {
                        return checkPos;
                    }
                }
            }
        }
    }
    
    return null;
}

/**
 * Find a valid reference block (used for placement)
 *
 * @param {Bot} bot - Mineflayer bot instance
 * @param {Vec3} targetPos - target placement position
 * @returns {Block|null} - the reference block found, or null
 */
function findValidReferenceBlock(bot, targetPos) {
    const faceVectors = [
        new Vec3(0, -1, 0),  // below
        new Vec3(0, 1, 0),   // above
        new Vec3(1, 0, 0),   // east
        new Vec3(-1, 0, 0),  // west
        new Vec3(0, 0, 1),   // south
        new Vec3(0, 0, -1)   // north
    ];
    
    for (const faceVector of faceVectors) {
        const refPos = targetPos.plus(faceVector);
        const refBlock = bot.blockAt(refPos);
        
        if (refBlock && refBlock.boundingBox === 'block') {
            return refBlock;
        }
    }
    
    return null;
}

/**
 * Find a diggable block (used to create space for placement)
 *
 * @param {Bot} bot - Mineflayer bot instance
 * @param {Vec3} center - search center (optional; defaults to bot position)
 * @returns {Block|null} - a diggable block, or null
 */
function findDiggableBlock(bot, center) {
    // P0: defensive check
    if (!center && !bot.entity) {
        console.warn("[findDiggableBlock] bot.entity is undefined and no center provided");
        return null;
    }
    const foot = center || bot.entity.position.floored();

    // List of blocks safe to dig
    const safeToDigBlocks = [
        'dirt', 'grass_block', 'sand', 'gravel', 
        'stone', 'cobblestone', 'andesite', 'diorite', 'granite',
        'oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log',
        'oak_leaves', 'birch_leaves', 'spruce_leaves', 'jungle_leaves', 'acacia_leaves', 'dark_oak_leaves',
        'tall_grass', 'fern', 'dead_bush'
    ];
    
    // Search blocks around the position
    const candidates = [
        foot.offset(1, 0, 0),
        foot.offset(-1, 0, 0),
        foot.offset(0, 0, 1),
        foot.offset(0, 0, -1),
        foot.offset(1, 1, 0),
        foot.offset(-1, 1, 0),
        foot.offset(0, 1, 1),
        foot.offset(0, 1, -1)
    ];
    
    for (const pos of candidates) {
        const block = bot.blockAt(pos);
        if (block && safeToDigBlocks.includes(block.name)) {
            // Check whether it's safe to dig (block above is not sand/gravel)
            const above = bot.blockAt(pos.offset(0, 1, 0));
            if (!above || !['sand', 'gravel'].includes(above.name)) {
                return block;
            }
        }
    }

    return null;
}

/**
 * Check whether a block can be placed at the specified position
 *
 * @param {Bot} bot - Mineflayer bot instance
 * @param {Vec3} pos - target position
 * @returns {boolean} - whether placement is possible
 */
function canPlaceAt(bot, pos) {
    const block = bot.blockAt(pos);
    if (!block || block.name !== 'air') return false;
    
    const refBlock = findValidReferenceBlock(bot, pos);
    return refBlock !== null;
}

/**
 * Move near the specified position
 *
 * @param {Bot} bot - Mineflayer bot instance
 * @param {Vec3} pos - target position
 * @param {number} range - approach distance
 */
async function moveNear(bot, pos, range = 2) {
    const goal = new GoalNear(pos.x, pos.y, pos.z, range);
    await gotoWithTimeout(bot, goal, 30000, `position (${Math.floor(pos.x)}, ${Math.floor(pos.y)}, ${Math.floor(pos.z)})`);
}

/**
 * Explore until a target is found
 *
 * @param {Bot} bot - Mineflayer bot instance
 * @param {Vec3} direction - exploration direction
 * @param {number} timeout - timeout in seconds
 * @param {Function} condition - stop-condition function; returns the found target or null
 * @returns {any} - the return value of the condition function
 */
async function exploreUntilFound(bot, direction, timeout, condition) {
    // P0: defensive check
    if (!bot.entity) {
        console.warn("[exploreUntilFound] bot.entity is undefined");
        return null;
    }

    const startTime = Date.now();
    const startPos = bot.entity.position.clone();

    while (Date.now() - startTime < timeout * 1000) {
        // P0: also check each iteration
        if (!bot.entity) {
            console.warn("[exploreUntilFound] bot.entity became undefined during exploration");
            return null;
        }

        // Check the condition
        const result = condition();
        if (result) return result;

        // Move in the specified direction
        const targetPos = bot.entity.position.plus(direction.scaled(10));
        try {
            await gotoWithTimeout(bot, new GoalNear(targetPos.x, targetPos.y, targetPos.z, 3), 15000, 'exploration target');
        } catch (e) {
            // If pathfinding fails or times out, try a different direction
            break;
        }

        // Rest briefly to avoid using too much CPU
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    return null;
}

/**
 * Find a block within an expanded range
 *
 * @param {Bot} bot - Mineflayer bot instance
 * @param {string} blockName - block name
 * @param {number} startRadius - starting search radius
 * @param {number} maxRadius - maximum search radius
 * @returns {Block|null} - the block found, or null
 */
function findBlockExpanded(bot, blockName, startRadius = 32, maxRadius = 64) {
    for (let r = startRadius; r <= maxRadius; r += 16) {
        const block = bot.findBlock({
            matching: b => b && b.name === blockName,
            maxDistance: r
        });
        if (block) return block;
    }
    return null;
}

/**
 * Safely dig a block (with retry and error handling)
 *
 * @param {Bot} bot - Mineflayer bot instance
 * @param {Block} block - block to dig
 * @param {number} maxRetries - maximum retry count
 * @returns {boolean} - whether successful
 */
async function safeDig(bot, block, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            await bot.dig(block);
            return true;
        } catch (e) {
            if (i === maxRetries - 1) {
                console.log(`[adaptive_helpers] safeDig failed after ${maxRetries} retries: ${e.message}`);
                return false;
            }
            // Wait briefly before retrying
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    }
    return false;
}

/**
 * Pillar up one block: place `item` at the bot's current foot cell by jumping to vacate
 * it and placing on the floor below, so the bot ascends onto the new block.
 *
 * Used to place a block AT a target the bot occupies (foot/head) without handing it to
 * GoalPlaceBlock (which towers-up + digs forever on an occupied target with scaffolding).
 * When the target is the head cell, one filler pillar raises the foot to the head, after
 * which it reduces to the foot case.
 *
 * @param {Bot} bot - Mineflayer bot instance
 * @param {Item} [item] - item to place; defaults to a scaffolding block from inventory
 *                        (bot.pathfinder.movements.getScaffoldingItem()).
 * @returns {Promise<void>} resolves once a block occupies the former foot cell and the bot
 *                          rose by one; throws if all attempts fail, or — when no `item` is
 *                          given — if no scaffolding block is available.
 */
async function pillarUp(bot, item) {
    const useScaffold = !item;
    const maxAttempts = 3;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        let filler = item;
        if (useScaffold) {
            const mv = bot.pathfinder && bot.pathfinder.movements;
            filler = (mv && typeof mv.getScaffoldingItem === "function")
                ? mv.getScaffoldingItem() : null;
        }
        if (!filler) {
            throw new Error("No scaffolding block available to pillar up");
        }
        const foot = bot.entity.position.floored();
        const below = bot.blockAt(foot.offset(0, -1, 0));
        await bot.equip(filler, "hand");
        bot.setControlState("jump", true);
        await bot.waitForTicks(7);   // airborne -> foot cell vacated
        try {
            await bot.placeBlock(below, new Vec3(0, 1, 0));   // place on the floor -> block at foot
        } catch (e) {
            // placeBlock can report a false negative while airborne; verified below
        }
        bot.setControlState("jump", false);
        await bot.waitForTicks(8);   // settle onto the new block
        const placed = bot.blockAt(foot);
        const rose = bot.entity.position.floored().y >= foot.y + 1;
        if (placed && placed.boundingBox === "block" && rose) {
            return;
        }
    }
    throw new Error(`Failed to pillar up after ${maxAttempts} attempts`);
}

// Note: this file is executed via eval() rather than require(); no module.exports needed
