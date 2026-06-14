// ============================================================
// Timeout Utilities for Pathfinder and CollectBlock Operations
// ============================================================
// These utilities prevent infinite blocking when pathfinder.goto()
// or collectBlock.collect() cannot reach the target.

/**
 * Generic timeout wrapper (with logging and cleanup)
 * @param {Promise} promise - the Promise to wrap
 * @param {number} ms - timeout in milliseconds
 * @param {string} errorMsg - error message on timeout
 * @returns {Promise} - the timeout-wrapped Promise
 */
function withTimeout(promise, ms, errorMsg) {
    let timeoutId;
    const timeoutPromise = new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
            console.log(`[Timeout] Operation timed out after ${ms/1000}s`);
            reject(new Error(errorMsg));
        }, ms);
    });
    return Promise.race([promise, timeoutPromise]).finally(() => clearTimeout(timeoutId));
}

/**
 * Get the bot position string (for diagnostics)
 * @param {Bot} bot - Mineflayer bot instance
 * @returns {string} - formatted position string
 */
function getBotPosStr(bot) {
    // P0: defensive check - bot.entity may be undefined on death/disconnect
    if (!bot.entity) {
        return "(unknown - bot.entity undefined)";
    }
    const pos = bot.entity.position;
    return `(${Math.floor(pos.x)}, ${Math.floor(pos.y)}, ${Math.floor(pos.z)})`;
}

/**
 * Extract a target position from a pathfinder Goal, across the known shapes.
 *   - GoalLookAtBlock / GoalBreakBlock: { pos: Vec3 }          -> full (x,y,z)
 *   - GoalBlock / GoalNear / GoalGetToBlock:  { x, y, z }      -> full (x,y,z)
 *   - GoalXZ / GoalNearXZ:                    { x, z }         -> (x, null, z)
 *   - GoalY:                                  { y }            -> (null, y, null)
 *   - Composite / Invert / Follow: unknown shape -> null
 * Returns { x, y, z } with individual coords possibly null, or null if unknown.
 */
function _extractGoalTarget(goal) {
    if (!goal || typeof goal !== 'object') return null;
    if (goal.pos && typeof goal.pos.x === 'number') {
        return { x: goal.pos.x, y: goal.pos.y, z: goal.pos.z };
    }
    const hasX = typeof goal.x === 'number';
    const hasY = typeof goal.y === 'number';
    const hasZ = typeof goal.z === 'number';
    if (hasX && hasY && hasZ) return { x: goal.x, y: goal.y, z: goal.z };
    if (hasX && hasZ) return { x: goal.x, y: null, z: goal.z };
    if (hasY) return { x: null, y: goal.y, z: null };
    return null;
}

/**
 * Diagnose why pathfinder.goto timed out. Mirrors diagnoseCollectFailure — probes
 * bot + target state at the moment of failure so the LLM can reason about the
 * actual cause (target high above through solid rock, bot in lava, target behind
 * a wall, etc.) instead of seeing only a generic timeout string. See Study 05
 * for architectural rationale.
 */
function diagnoseGotoFailure(bot, goal, targetDescription) {
    const parts = [];
    if (!bot.entity) return 'bot.entity undefined (died / disconnected)';
    const pos = bot.entity.position;
    parts.push(`bot at (${pos.x.toFixed(2)},${pos.y.toFixed(2)},${pos.z.toFixed(2)})`);

    const target = _extractGoalTarget(goal);
    if (target) {
        const tx = target.x, ty = target.y, tz = target.z;
        const coord = `(${tx !== null ? tx : '?'},${ty !== null ? ty : '?'},${tz !== null ? tz : '?'})`;
        parts.push(`target ${targetDescription} at ${coord}`);
        if (tx !== null && ty !== null && tz !== null) {
            const dx = tx - pos.x, dy = ty - pos.y, dz = tz - pos.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            parts.push(`distance ${dist.toFixed(1)} blocks (dy=${dy.toFixed(1)})`);
            try {
                // Use pos.offset(...).floored() to construct Vec3-shaped probes
                // without needing to require('vec3') (which fails when this file
                // is eval'd from another cwd in tests / reproducers).
                const targetVec = pos.offset(dx, dy, dz).floored();
                const actual = bot.blockAt(targetVec);
                parts.push(`block at target: ${actual ? actual.name : 'null'}`);
                // Sample 3 points along the straight line from bot to target.
                // If many of these are solid (non-air), the path is likely blocked
                // by terrain that pathfinder must dig/climb through.
                const between = [];
                for (let i = 1; i <= 3; i++) {
                    const t = i / 4;
                    const b = bot.blockAt(pos.offset(dx * t, dy * t, dz * t).floored());
                    if (b && b.name !== 'air' && b.name !== 'cave_air' && b.name !== 'void_air') {
                        between.push(b.name);
                    }
                }
                if (between.length) parts.push(`straight-line crosses ${between.join(',')}`);
            } catch (_) { /* offset / blockAt probing failed; skip */ }
        }
    } else {
        parts.push(`goal shape unknown (target=${targetDescription})`);
    }

    const conds = [];
    if (bot.entity.isInLava) conds.push('in_lava');
    if (bot.entity.isInWater) conds.push('in_water');
    if (!bot.entity.onGround) conds.push('airborne');
    if (bot.oxygenLevel != null && bot.oxygenLevel < 5) conds.push(`low_oxygen(${bot.oxygenLevel})`);
    if (bot.health != null && bot.health < 10) conds.push(`low_health(${bot.health})`);
    if (conds.length) parts.push(`bot state: ${conds.join(',')}`);
    parts.push(`held: ${bot.heldItem ? bot.heldItem.name : 'empty'}`);
    return parts.join('; ');
}

/**
 * pathfinder.goto-specific timeout (auto-clears goal, provides friendly error message + diagnostics)
 * @param {Bot} bot - Mineflayer bot instance
 * @param {Goal} goal - pathfinder goal
 * @param {number} timeoutMs - timeout in milliseconds, default 30000
 * @param {string} targetDescription - target description (for error message)
 */
async function gotoWithTimeout(bot, goal, timeoutMs = 30000, targetDescription = 'target') {
    const baseMsg = `Movement to ${targetDescription} timed out after ${timeoutMs/1000}s. `;
    const genericMsg = baseMsg +
        `Bot position: ${getBotPosStr(bot)}. ` +
        `The path may be blocked or unreachable. ` +
        `Suggestions: 1) Move to a different position first, 2) Check if there are obstacles, ` +
        `or 3) Try a different approach to reach ${targetDescription}.`;
    try {
        await withTimeout(bot.pathfinder.goto(goal), timeoutMs, genericMsg);
    } catch (err) {
        bot.pathfinder.setGoal(null);
        // PATCH (skillnet, P3 extension): on timeout, probe state and enrich
        // the error with structured diagnostic info so the LLM can reason about
        // the actual cause (target 19 blocks above, path blocked by stone, bot
        // stuck in lava, etc.). Same architectural pattern as collectWithTimeout.
        if (err && err.message && err.message.startsWith('Movement to ')) {
            const diag = diagnoseGotoFailure(bot, goal, targetDescription);
            throw new Error(baseMsg + diag);
        }
        throw err;
    }
}

/**
 * Diagnose why a mineBlock/collect operation timed out — probe the bot + target
 * state at the moment of failure so the LLM can reason about the actual cause
 * (stuck in lava, target unreachable through rock, distance too large, wrong
 * tool, etc.) instead of seeing only a generic timeout string. See Study 05
 * (docs/projectwebpage/findings/05-chest-blocked-by-solid-above.md) for the
 * architectural rationale: primitives must surface mechanistic state on failure.
 */
function diagnoseCollectFailure(bot, targets, targetName) {
    const out = [];
    if (!bot.entity) return 'Bot state: entity undefined (died / disconnected).';
    const pos = bot.entity.position;
    out.push(`Bot at (${Math.floor(pos.x)},${Math.floor(pos.y)},${Math.floor(pos.z)}).`);
    const conds = [];
    if (bot.entity.isInLava) conds.push('in_lava');
    if (bot.entity.isInWater) conds.push('in_water');
    if (!bot.entity.onGround) conds.push('airborne');
    if (bot.oxygenLevel != null && bot.oxygenLevel < 5) conds.push(`low_oxygen(${bot.oxygenLevel})`);
    if (bot.health != null && bot.health < 10) conds.push(`low_health(${bot.health})`);
    if (conds.length) out.push(`Bot state: ${conds.join(', ')}.`);

    // PATCH (skillnet): pick the candidate the bot is CURRENTLY closest to
    // AND whose blockAt still matches its cached name. Previous version
    // always dumped targets[0] — the closest-at-findBlocks-time candidate —
    // which may have been mined long before the wrapper fired (the bot's
    // own successful mine of an early target leaves that target's position
    // as air; if targets[0] was that early target, the "Block at target
    // has changed: air" line is misleading — bot is NOT stuck on it). The
    // heuristic below identifies the target collectblock is actively
    // attempting at the moment of timeout.
    let stuckTarget = null;
    if (Array.isArray(targets) && targets.length > 0) {
        let minDist = Infinity;
        for (const t of targets) {
            if (!t || !t.position) continue;
            const cur = bot.blockAt(t.position);
            // Already-mined candidates (block name changed) are not what
            // collectblock is stuck on — skip them when picking stuckTarget.
            if (!cur || cur.name !== t.name) continue;
            const dx = t.position.x - pos.x;
            const dy = t.position.y - pos.y;
            const dz = t.position.z - pos.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < minDist) { minDist = dist; stuckTarget = t; }
        }
        // Fallback: if every candidate has been mined / changed, fall back
        // to targets[0] so the diag still produces SOMETHING. Tagging with
        // the "(legacy fallback — all candidates were mined or changed)"
        // hint lets readers distinguish this case from a real stuck-target.
        if (stuckTarget == null) {
            stuckTarget = targets[0];
            if (stuckTarget && stuckTarget.position) {
                const tp = stuckTarget.position;
                out.push(`First target: ${stuckTarget.name} at (${tp.x},${tp.y},${tp.z}) (legacy fallback — all candidates were mined or changed).`);
            }
            stuckTarget = null; // signal "do not run the normal block below"
        }
    } else if (targets && targets.position) {
        stuckTarget = targets;
    }
    if (stuckTarget && stuckTarget.position) {
        const tp = stuckTarget.position;
        out.push(`Stuck target: ${stuckTarget.name} at (${tp.x},${tp.y},${tp.z}).`);
        const above = bot.blockAt(tp.offset(0, 1, 0));
        if (above) out.push(`Above target: ${above.name}.`);
        const below = bot.blockAt(tp.offset(0, -1, 0));
        if (below) out.push(`Below target: ${below.name}.`);
        const dx = tp.x - pos.x, dy = tp.y - pos.y, dz = tp.z - pos.z;
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
        out.push(`Distance: ${dist.toFixed(1)} blocks (dy=${dy.toFixed(1)}).`);
    }
    out.push(`Held item: ${bot.heldItem ? bot.heldItem.name : 'empty-hand'}.`);
    return out.join(' ');
}

/**
 * collectBlock.collect-specific timeout (dynamically computed, provides friendly error message + diagnostics)
 * @param {Bot} bot - Mineflayer bot instance
 * @param {Block|Block[]|Entity|Entity[]} targets - targets to collect
 * @param {Object} options - collectBlock options
 * @param {number} count - target count (used to compute timeout and error message)
 * @param {string} targetName - target name (for error message)
 */
async function collectWithTimeout(bot, targets, options, count = 1, targetName = 'blocks') {
    // PATCH (skillnet): generous per-candidate budget + hard cap. The earlier
    // formula Math.max(30, count*15) was too tight: legitimately-reachable
    // candidates that require pathfinder to dig ~20-30 stone blocks to
    // approach take ~25-35s each, exceeding the budget. dig_cave reproducer
    // shows bot reaching y=51 (1 block from a y=50 target) at 30s elapsed —
    // wrapper fires while bot is moments from success. Bumped to 30s per
    // candidate with a 60s floor (gives margin for the common case where
    // most candidates are easy), and a 5-minute HARD_CAP that prevents
    // pathfinder-state-machine bugs or stalled bot.dig promises from
    // hanging the /step call beyond PSN's outer 900s request_timeout.
    const HARD_CAP_MS = 300000;  // 5 minutes
    const timeoutMs = Math.min(HARD_CAP_MS, Math.max(60000, count * 30000));
    const baseMsg = `Collecting ${count} ${targetName} timed out after ${timeoutMs/1000}s. `;
    const genericMsg = baseMsg +
        `Bot position: ${getBotPosStr(bot)}. ` +
        `The bot may be stuck or the ${targetName} are unreachable. ` +
        `Suggestions: 1) Use exploreUntil() to find ${targetName} in a different area, ` +
        `2) Try collecting targets that are closer or more accessible, ` +
        `or 3) Move to a different location and retry.`;
    try {
        await withTimeout(bot.collectBlock.collect(targets, options), timeoutMs, genericMsg);
    } catch (err) {
        // PATCH (skillnet): a timed-out race does NOT stop the wrapped
        // collect(): the abandoned loop kept mining, swapping the held item
        // and seizing the pathfinder during the FOLLOWING skill actions and
        // steps, making their effects unattributable. Cancel the task and
        // give the loop a bounded grace to confirm exit (collectBlock's
        // cancellation is cooperative; the grace covers an in-flight dig).
        // On a non-timeout rejection the task already cleared its targets,
        // so cancelTask is a no-op there.
        try {
            if (bot.collectBlock) {
                let graceTimer;
                await Promise.race([
                    bot.collectBlock.cancelTask(),
                    new Promise((r) => { graceTimer = setTimeout(r, 15000); }),
                ]).finally(() => clearTimeout(graceTimer));
            }
        } catch (e) {}
        bot.pathfinder.setGoal(null);
        // PATCH (skillnet, P3): on timeout, probe state and enrich the error
        // with structured diagnostic info so the LLM can reason about the
        // actual cause. Fall back to the generic message if the error isn't
        // our own timeout (e.g., cancellation, unexpected exception).
        if (err && err.message && err.message.startsWith('Collecting ')) {
            const diag = diagnoseCollectFailure(bot, targets, targetName);
            throw new Error(baseMsg + diag);
        }
        throw err;
    }
}
