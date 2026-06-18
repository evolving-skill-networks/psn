'use strict';
/*
 * PSN surface-for-air watcher: an always-on survival reflex that stops the bot
 * drowning underwater.
 *
 * Why this exists separately from the pathfinder's own in-tick breath guard:
 * the pathfinder guard only runs while the bot is ACTIVELY pathfinding. When
 * the bot sits underwater NOT pathfinding -- idle, digging a block in place, or
 * having just REACHED a submerged goal -- that guard is dormant and the bot
 * drowns. Confirmed on a live LAN world: needAir stayed undefined, isMoving
 * false, health 20 -> drowned. This physicsTick reflex closes that gap.
 *
 * Behaviour: whenever the bot's head is submerged and air is low AND the
 * pathfinder is NOT driving, take control to swim straight up for air, then
 * release once air is restored. Air is read from entity metadata key 1 (raw air
 * ticks ~0..300), which is reliable; bot.oxygenLevel is NOT (it intermittently
 * reads bogus high values). Hysteresis (surface <= AIR_LOW, resume >= AIR_OK)
 * guarantees a full breath instead of a partial 1-tick breath that re-drowns.
 *
 * Only saves the bot when air is reachable above (the common aquifer / open-top
 * case). A fully-enclosed water ceiling with no reachable air cannot be helped
 * here. Every block is try/caught so the reflex can never throw into the physics
 * loop. Shared by the mineflayer bridge (index.js) and the regression scripts so
 * both exercise identical logic.
 */

// Air thresholds, SHARED with the pathfinder's in-tick breath guard. Both write
// the SAME need flag (bot._pfNeedAir) and use the SAME thresholds, so the
// hysteresis state carries seamlessly across moving<->idle handoffs (separate
// flags dropped a breath when an integrated/LAN server's tick lagged during the
// toggle). Surfacing at 160 (~8s of air left) leaves a wide margin for tick lag.
const AIR_LOW = 160;   // raw air ticks: start surfacing below this
const AIR_OK = 240;    // raw air ticks: only stop surfacing above this (full breath)

function installBreathWatcher(bot, opts) {
  const log = (opts && opts.log) || (() => {});
  if (bot._psnBreathWatcherInstalled) return;
  bot._psnBreathWatcherInstalled = true;

  // releaseHold() drops the surfacing controls exactly once, so we never leave
  // them stuck true after a breath (a bug that silently floats the bot).
  const releaseHold = () => {
    if (bot._psnHolding) {
      bot.setControlState('jump', false);
      bot.setControlState('forward', false);
      bot.setControlState('sprint', false);
      bot._psnHolding = false;
    }
  };

  bot.on('physicsTick', () => {
    try {
      if (!bot.entity || !bot.entity.isInWater) { bot._pfNeedAir = false; releaseHold(); return; }

      // This guard runs EVERY tick regardless of pathfinder state. It only
      // overrides controls when air is actually low (below it does nothing and
      // never touches controls), so it does not fight normal navigation -- but
      // it DOES cover the case the in-tick guard misses: when the bot has
      // reached a deep submerged goal and the pathfinder reports isMoving=true
      // yet has stopped driving, the in-tick breath logic no longer surfaces it.
      // Without this always-on guard the bot sits at the bottom and drowns.
      //
      // While the in-tick guard IS actively driving breath (fresh stamp), defer
      // to it so we do not double-drive the controls and erode an active descent.
      // We take over only once that stamp goes stale (it stopped running).
      if (bot._pfInTickTs && (Date.now() - bot._pfInTickTs) < 150) { bot._psnHolding = false; return; }

      const md = bot.entity.metadata;
      const air = (md && typeof md[1] === 'number') ? md[1] : null;
      // use the live eye height (0.4 when sprint-swimming, 1.62 standing) so the
      // head-in-water check matches where the bot actually breathes
      const eye = bot.blockAt(bot.entity.position.offset(0, bot.entity.eyeHeight || 1.62, 0));
      const headInWater = !!(eye && eye.name === 'water');

      // update the need flag (hysteresis) ONLY from real air data; a clear head
      // means we are breathing, so the need is satisfied. We intentionally do
      // NOT treat a momentarily-null air reading as "low" -- that misfires on
      // the first tick after a teleport and is not a real low-air condition.
      if (!headInWater) bot._pfNeedAir = false;
      else if (air != null) {
        if (air <= AIR_LOW) bot._pfNeedAir = true;
        else if (air >= AIR_OK) bot._pfNeedAir = false;
      }

      if (bot._pfNeedAir && headInWater) {
        // sprint-swim straight UP for air (fast). The pathfinder's in-tick guard
        // does the same while navigating; matching it here means an idle bot that
        // has settled deep surfaces just as fast and does not erode the margin.
        try { bot.look(bot.entity.yaw, Math.PI / 2, true); } catch (e) {}
        bot.setControlState('forward', true);
        bot.setControlState('jump', true);
        bot.setControlState('sneak', false);
        bot.setControlState('sprint', true);
        bot._psnHolding = true;
      } else {
        releaseHold();
      }
    } catch (e) { /* never let the reflex throw into the tick */ }
  });
  bot.once('end', () => { try { bot._psnAirNeed = false; } catch (e) {} });
  log('[PSN BreathWatcher] installed for ' + (bot.username || '?'));
}

module.exports = { installBreathWatcher, AIR_LOW, AIR_OK };
