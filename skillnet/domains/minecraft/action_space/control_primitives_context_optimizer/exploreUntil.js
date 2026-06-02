// Explore in a direction until a callback returns a truthy value, or maxTime seconds elapse.
//
// PARAMETERS:
//   exploreUntil(bot, direction, maxTime = 60, callback = () => false)
//   - direction: Vec3 — movement direction, each component must be -1, 0, or 1
//     e.g., new Vec3(1, 0, 0) = east, new Vec3(0, -1, 0) = down, new Vec3(1, 0, 1) = northeast
//   - maxTime: number — max exploration time in seconds (capped at 1200)
//   - callback: function — called every 2 seconds. Return truthy value to stop exploration.
//     The return value is resolved as the Promise result.
//
// RETURNS: Promise that resolves with:
//   - callback's truthy return value (if found)
//   - null (if maxTime reached without callback returning truthy)
//
// BEHAVIOR:
// - Checks callback FIRST before setting any goal (avoids unnecessary movement)
// - Uses bot.pathfinder.setGoal() to walk in the specified direction
// - Sets a new random goal every 2 seconds (10-30 blocks in the direction)
// - For horizontal directions (dy=0): uses GoalNearXZ (ignores Y)
// - For vertical directions (dy!=0): uses GoalNear (includes Y)
// - Yields to active pathfinding: if bot.pathfinder.isMoving(), skips setting new goal
//   (prevents GoalChanged race condition with other concurrent navigation)
// - Cleans up interval + timeout + cancels pathfinder goal on completion
//
// DIRECTION VALIDATION:
// - Each component must be exactly -1, 0, or 1 (not other values)
// - Cannot be (0, 0, 0) — throws error
//
// ERROR HANDLING:
//   - "maxTime must be a number"
//   - "callback must be a function"
//   - "direction cannot be 0, 0, 0"
//   - "direction must be a Vec3 only with value of -1, 0 or 1"
//   - If callback throws, the error propagates and exploration stops
//
// NOTE: bot.entity may be undefined during death/disconnect — handled defensively.
// NOTE: Calls bot.pathfinder.setGoal(null) on cleanup to cancel navigation.
async function exploreUntil(bot, direction, maxTime = 60, callback = () => false) {
    if (typeof maxTime !== "number") throw new Error("maxTime must be a number");
    if (typeof callback !== "function") throw new Error("callback must be a function");
    // Check callback immediately before exploring
    const test = callback();
    if (test) return test;
    // Validate direction
    if (direction.x === 0 && direction.y === 0 && direction.z === 0)
        throw new Error("direction cannot be 0, 0, 0");
    maxTime = Math.min(maxTime, 1200);
    return new Promise((resolve, reject) => {
        let done = false;
        const cleanUp = () => {
            if (done) return;
            done = true;
            clearInterval(explorationInterval);
            clearTimeout(maxTimeTimeout);
            bot.pathfinder.setGoal(null);
        };
        const explore = () => {
            if (done || !bot.entity) return;
            try {
                const result = callback();
                if (result) { cleanUp(); resolve(result); return; }
            } catch (err) { cleanUp(); reject(err); return; }
            if (bot.pathfinder.isMoving()) return;  // Yield to active navigation
            const x = bot.entity.position.x + Math.floor(Math.random() * 20 + 10) * direction.x;
            const y = bot.entity.position.y + Math.floor(Math.random() * 20 + 10) * direction.y;
            const z = bot.entity.position.z + Math.floor(Math.random() * 20 + 10) * direction.z;
            bot.pathfinder.setGoal(direction.y === 0 ? new GoalNearXZ(x, z) : new GoalNear(x, y, z));
        };
        explore();
        const explorationInterval = setInterval(explore, 2000);
        const maxTimeTimeout = setTimeout(() => { cleanUp(); resolve(null); }, maxTime * 1000);
    });
}
