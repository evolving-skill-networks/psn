const fs = require("fs");
const express = require("express");
const bodyParser = require("body-parser");
const mineflayer = require("mineflayer");

const skills = require("./lib/skillLoader");
const { initCounter, getNextTime } = require("./lib/utils");
const obs = require("./lib/observation/base");
const OnChat = require("./lib/observation/onChat");
const OnError = require("./lib/observation/onError");
const { Voxels, BlockRecords } = require("./lib/observation/voxels");
const Status = require("./lib/observation/status");
const Inventory = require("./lib/observation/inventory");
const OnSave = require("./lib/observation/onSave");
const Chests = require("./lib/observation/chests");
const Spatial = require("./lib/observation/spatial");
const BlockActions = require("./lib/observation/blockActions");
const { plugin: tool } = require("mineflayer-tool");
// Hoisted to module top so the /start handler can attach the 'raw' listener
// synchronously, right after `mineflayer.createBot()` returns and before
// the LOGIN handshake packets arrive on the next event-loop tick.
const McprRecorder = require('./mcpr_recorder');
// IIFE-wrap each top-level skill function so that LLM-written module-top
// declarations (e.g. `let mcData = null;`) can't collide with globalDepsCode's
// `var mcData` or with declarations in other skill files.
const { wrapProgramsInIIFE } = require('./iife_wrap');

let bot = null;
let mcprRecorder = null;

// PATCH (skillnet): step-boundary quarantine. The per-step otherError handler
// only exists inside the /step window, so a background exception landing
// BETWEEN steps used to crash the whole bridge process (and an unhandled
// promise rejection crashed it even during a step -- nothing listened). Both
// are now contained: logged, queued, and surfaced into the NEXT step's
// observations via the onError channel (queuing instead of emitting matters:
// an immediate emit between steps would be drained into cumulativeObs and
// wiped at the next step's reset, and an emit during a /start connection
// window would be mistaken for a connection failure).
let stepInFlight = false;
let stepEpoch = 0;
const pendingBridgeErrors = [];

function surfaceBridgeError(msg) {
    if (stepInFlight && bot && typeof bot.event === "function") {
        try { bot.emit("error", new Error(msg)); return; } catch (e) {}
    }
    pendingBridgeErrors.push(msg);
    if (pendingBridgeErrors.length > 20) pendingBridgeErrors.shift();
}

process.on("uncaughtException", (err) => {
    // All listeners fire, so deferring here can never block the per-step
    // otherError; the listener count also covers the window where otherError
    // was already removed but the step is still settling.
    if (stepInFlight && process.listenerCount("uncaughtException") > 1) return;
    console.error("[Bridge] Uncaught exception (no step owner):", (err && err.stack) || err);
    surfaceBridgeError(`[between-steps] uncaught exception: ${(err && err.message) || err}`);
});

process.on("unhandledRejection", (reason) => {
    console.error(`[Bridge] Unhandled rejection ${stepInFlight ? "during step" : "between steps"}:`,
        (reason && reason.stack) || reason);
    surfaceBridgeError(`[background] unhandled rejection: ${(reason && reason.message) || reason}`);
});

// Clear every actuator a skill can leave running (pathfinder goal, collect
// task, dig, pvp attack, control states) so nothing keeps acting across step
// boundaries and the next step's effects stay attributable to its own code.
function quarantineActuators() {
    try { if (bot && bot.collectBlock) bot.collectBlock.cancelTask().catch(() => {}); } catch (e) {}
    try { if (bot && bot.pathfinder) bot.pathfinder.setGoal(null); } catch (e) {}
    try { if (bot && bot.stopDigging) bot.stopDigging(); } catch (e) {}
    try { if (bot && bot.pvp && bot.pvp.target && bot.pvp.forceStop) bot.pvp.forceStop(); } catch (e) {}
    try { if (bot && bot.clearControlStates) bot.clearControlStates(); } catch (e) {}
}

const app = express();

app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ limit: "50mb", extended: false }));

app.post("/start", (req, res) => {
    if (bot) onDisconnect("Restarting bot");
    bot = null;
    console.log(req.body);
    
    // Set response timeout (60s; gives the Minecraft server more time)
    const responseTimeout = setTimeout(() => {
        if (!res.headersSent) {
            console.error("[Start] Response timeout: bot failed to connect and spawn within 60 seconds");
            console.error(`[Start] Target server: localhost:${req.body.port}`);
            if (bot) {
                console.error("[Start] Bot status:", {
                    connected: bot.connected,
                    entity: bot.entity ? "exists" : "missing",
                    username: bot.username
                });
                try {
                    bot.end();
                } catch (e) {
                    console.error("[Start] Error while closing bot:", e);
                }
                bot = null;
            }
            res.status(500).json({
                error: "Bot connection timeout: failed to spawn within 60 seconds",
                message: "Bot connection timeout: failed to spawn within 60 seconds",
                port: req.body.port,
                suggestions: [
                    "Make sure the Minecraft server is running",
                    "Make sure LAN is open (Open to LAN)",
                    "Make sure the server is not paused (press F3+P to unpause)",
                    "Check firewall settings"
                ]
            });
        }
    }, 60000);

    // Defensive recovery for the "MC stuck paused" failure mode. When a
    // previous run crashed during reset between sending /pause
    // (server paused + sync save) and killing mineflayer, MC can be left
    // paused. The next bot's login+JoinGame succeed (those are protocol
    // events, not world ticks) but the spawn position update never
    // arrives because the world isn't ticking. If 30s pass and we passed
    // JoinGame (bot.entity exists) but no spawn, send /pause to toggle
    // MC back to running. Normal spawn happens in 1-3 seconds, so 30s
    // has wide margin against accidentally pausing a slow-but-running
    // server.
    const earlyRecoveryTimeout = setTimeout(() => {
        if (responseSent || !bot) return;
        if (!bot.entity) return;
        console.warn(
            "[Start] spawn delayed >30s after JoinGame; attempting /pause " +
            "toggle in case MC server is stuck paused"
        );
        try {
            bot.chat("/pause");
        } catch (e) {
            console.warn("[Start] recovery /pause chat failed:", e.message);
        }
    }, 30000);

    let responseSent = false;
    const sendResponse = (data, statusCode = 200) => {
        if (!responseSent && !res.headersSent) {
            responseSent = true;
            clearTimeout(responseTimeout);
            clearTimeout(earlyRecoveryTimeout);
            if (statusCode === 200) {
                res.json(data);
            } else {
                res.status(statusCode).json(data);
            }
        }
    };
    
    // Validate request parameters
    if (!req.body || !req.body.port) {
        console.error("[Start] Missing required parameter: port");
        sendResponse({ error: "Missing required parameter: port" }, 400);
        return;
    }
    
    try {
        bot = mineflayer.createBot({
            host: "localhost", // minecraft server ip
            port: req.body.port, // minecraft server port
            username: "bot",
            disableChatSigning: true,
            checkTimeoutInterval: 60 * 60 * 1000,
        });
        // Begin buffering raw packets IMMEDIATELY so we capture the LOGIN
        // handshake. ReplayMod's replaystudio reader starts in LOGIN state
        // for fileFormatVersion >= 14 and only transitions to PLAY when it
        // sees a LoginSuccess packet — so a recording that starts at PLAY
        // is unparseable. We attach the listener synchronously here, before
        // the event loop yields to the connection.
        try {
            mcprRecorder = new McprRecorder(bot);
            mcprRecorder.startBuffering();
        } catch (e) {
            console.error("[Start] McprRecorder.startBuffering failed:", e);
            mcprRecorder = null;
        }
        console.log(`[Start] Connecting to Minecraft server localhost:${req.body.port}`);
    } catch (e) {
        console.error("[Start] Failed to create bot:", e);
        console.error("[Start] Error stack:", e.stack);
        sendResponse({ 
            error: e.message || String(e),
            stack: e.stack 
        }, 400);
        return;
    }
    
    // Listen for connection events
    bot.once("connect", () => {
        console.log(`[Start] Bot connected to Minecraft server localhost:${req.body.port}`);
    });
    
    bot.once("error", onConnectionFailed);
    
    // Listen for connection errors
    bot.on("error", (err) => {
        console.error("[Start] Bot error:", err);
        console.error(`[Start] Error code: ${err.code || "N/A"}`);
        console.error(`[Start] Error message: ${err.message || String(err)}`);
        // If no response has been sent yet and this is not a connection-failure error (handled by onConnectionFailed)
        if (!responseSent && err.code !== "ECONNREFUSED" && err.code !== "ETIMEDOUT") {
            // Wait briefly to see if onConnectionFailed is triggered
            setTimeout(() => {
                if (!responseSent) {
                    console.error("[Start] Error after bot connected; sending error response");
                    sendResponse({ 
                        error: err.message || String(err),
                        code: err.code || "UNKNOWN"
                    }, 400);
                }
            }, 1000);
        }
    });

    // Event subscriptions
    bot.waitTicks = req.body.waitTicks;
    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];
    bot.iron_pickaxe = false;

    // Death/respawn event handling - for diagnostics and state reset
    function _posDump(tag) {
        try {
            const e = bot.entity;
            const p = e && e.position;
            const v = e && e.velocity;
            console.log(
                `[Diag ${tag}] pos=(${p?.x},${p?.y},${p?.z}) ` +
                `vel=(${v?.x},${v?.y},${v?.z}) ` +
                `onGround=${e?.onGround} health=${bot.health} ` +
                `dead=${bot.entity?.metadata?.[6]} ` +
                `t=${Date.now()}`
            );
        } catch (e) {
            console.log(`[Diag ${tag}] pos dump failed: ${e.message}`);
        }
    }
    bot._posDump = _posDump;
    // Sliding window of recent position/velocity snapshots. When the first
    // NaN is detected, the entire window is dumped so we can see exactly
    // what move sequence preceded the NaN — the previous trace only logged
    // a single "first NaN" line with no upstream context.
    const _NAN_WINDOW = 80;
    const _recentMoves = [];
    let _nanWarned = false;
    bot.on("move", () => {
        try {
            const p = bot.entity && bot.entity.position;
            const v = bot.entity && bot.entity.velocity;
            const snap = {
                t: Date.now(),
                x: p?.x, y: p?.y, z: p?.z,
                vx: v?.x, vy: v?.y, vz: v?.z,
                onGround: bot.entity?.onGround,
            };
            _recentMoves.push(snap);
            if (_recentMoves.length > _NAN_WINDOW) _recentMoves.shift();
            if (!_nanWarned && p &&
                (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z))) {
                _nanWarned = true;
                console.error(
                    `[Diag NaN] FIRST NaN position detected: ` +
                    `(${p.x},${p.y},${p.z}) t=${Date.now()}`
                );
                console.error(
                    `[Diag NaN] Sliding window of last ${_recentMoves.length} ` +
                    `move events (oldest → newest):`
                );
                for (const m of _recentMoves) {
                    console.error(
                        `[Diag NaN]   t=${m.t} pos=(${m.x},${m.y},${m.z}) ` +
                        `vel=(${m.vx},${m.vy},${m.vz}) onGround=${m.onGround}`
                    );
                }
                console.error(
                    `[Diag NaN] JS stack at first-NaN detection:\n` +
                    new Error().stack
                );
            }
        } catch (_) {}
    });

    bot.on("death", () => {
        console.log("[Event] Bot died! Resetting stuck detection state...");
        _posDump("death");
        bot.stuckTickCounter = 0;
        bot.stuckPosList = [];
    });

    bot.on("respawn", () => {
        console.log("[Event] Bot respawned");
        _posDump("respawn");
        bot.stuckTickCounter = 0;
        bot.stuckPosList = [];
        // Re-apply night vision (cleared on death/respawn).
        applyNightVision(bot);
    });

    // Permanent night vision for the bot. Pure client-side rendering
    // effect — has zero impact on mining/pathfinding logic (those run on
    // server-side block data) — used so mcpr renders show interior cave
    // blocks during underground sequences instead of pitch black.
    // Hidden particle flag (`true`) keeps the visual clean for video.
    // Off-switch via env var if a run shouldn't have it.
    function applyNightVision(b) {
        if (process.env.BOT_NIGHT_VISION === "0") return;
        try {
            b.chat("/effect give @s minecraft:night_vision 999999 0 true");
        } catch (e) {
            console.warn("[NightVision] failed:", e.message);
        }
    }

    bot.on("health", () => {
        if (bot.health < 5) {
            console.warn(`[Health] Low health warning: ${bot.health}/20`);
        }
    });

    // Implement waitForTicks method
    bot.waitForTicks = function(ticks) {
        return new Promise((resolve) => {
            if (ticks <= 0) {
                resolve();
                return;
            }
            let count = 0;
            let resolved = false;
            const doResolve = () => {
                if (resolved) return;
                resolved = true;
                bot.removeListener("physicsTick", tickHandler);
                clearTimeout(timeout);
                resolve();
            };
            
            const tickHandler = () => {
                count++;
                if (count >= ticks) {
                    doResolve();
                }
            };
            
            // Add timeout to prevent infinite waiting
            // Use a more aggressive timeout: max 5 seconds regardless of ticks
            const timeoutMs = Math.min(ticks * 50 + 1000, 5000);
            const timeout = setTimeout(() => {
                console.log(`Warning: waitForTicks(${ticks}) timed out after ${timeoutMs}ms, resolving anyway`);
                doResolve();
            }, timeoutMs);
            
            bot.on("physicsTick", tickHandler);
        });
    };

    bot.on("kicked", (reason) => {
        console.error("[Start] Bot was kicked:", reason);
        _posDump("kicked");
        // Force-commit the buffered packet stream so a later post-mortem
        // can read the raw position packets MC rejected. Without this the
        // SIGTERM path discards the buffer.
        try {
            if (mcprRecorder && typeof mcprRecorder.commit === "function") {
                const debugOut = `/tmp/mineflayer_kick_debug_${Date.now()}.mcpr`;
                mcprRecorder.commit(debugOut);
                console.error(`[Diag kicked] mcpr force-committed → ${debugOut}`);
            }
        } catch (e) {
            console.error(`[Diag kicked] mcpr commit failed: ${e.message}`);
        }
        onDisconnect(`Bot kicked: ${reason}`);
        if (!responseSent) {
            sendResponse({ error: `Bot was kicked: ${reason}` }, 400);
        }
    });

    // mounting will cause physicsTick to stop
    bot.on("mount", () => {
        bot.dismount();
    });

    // Track if setup is complete to avoid double execution
    let setupComplete = false;
    let itemTicks = 1; // Initialize itemTicks in outer scope
    
    const completeSetup = async (ticks) => {
        if (setupComplete) return;
        setupComplete = true;
        
        try {
            // if iron_pickaxe is in bot's inventory
            if (
                bot.inventory.items().find((item) => item.name === "iron_pickaxe")
            ) {
                bot.iron_pickaxe = true;
            }

            const { pathfinder } = require("mineflayer-pathfinder");
            const tool = require("mineflayer-tool").plugin;
            const collectBlock = require("mineflayer-collectblock").plugin;
            const pvp = require("mineflayer-pvp").plugin;
            const minecraftHawkEye = require("minecrafthawkeye").default;
            bot.loadPlugin(pathfinder);
            bot.loadPlugin(tool);
            bot.loadPlugin(collectBlock);
            bot.loadPlugin(pvp);
            bot.loadPlugin(minecraftHawkEye);

            // bot.collectBlock.movements.digCost = 0;
            // bot.collectBlock.movements.placeCost = 0;

            obs.inject(bot, [
                OnChat,
                OnError,
                Voxels,
                Status,
                Inventory,
                OnSave,
                Chests,
                BlockRecords,
                Spatial,
                BlockActions,
            ]);
            skills.inject(bot);

            if (req.body.spread) {
                bot.chat(`/spreadplayers ~ ~ 0 300 under 80 false @s`);
                await bot.waitForTicks(bot.waitTicks);
            }

            await bot.waitForTicks(bot.waitTicks * ticks);
            
            // Check whether bot is still valid
            if (!bot || !bot.entity) {
                throw new Error("Bot disconnected before setup completion");
            }
            
            try {
                const obsResult = bot.observe();
                sendResponse(obsResult);
            } catch (obsErr) {
                console.error(`[Start] Error in bot.observe(): ${obsErr.message}`);
                sendResponse({
                    error: 'Failed to observe bot state',
                    message: obsErr.message
                }, 500);
            }

            // Recheck whether bot is still valid
            if (bot && bot.entity) {
                initCounter(bot);
                bot.chat("/gamerule keepInventory true");
                bot.chat("/gamerule doDaylightCycle false");
            }
        } catch (err) {
            console.error("[Start] completeSetup error:", err);
            console.error("[Start] Error stack:", err.stack);
            if (!responseSent) {
                sendResponse({ 
                    error: err.message || String(err),
                    stack: err.stack 
                }, 500);
            }
        }
    };
    
    bot.once("spawn", async () => {
        console.log("[Start] Bot spawned successfully");
        if (bot._posDump) bot._posDump("spawn-entry");
        bot.removeListener("error", onConnectionFailed);
        let itemTicks = 1;

        try {
            // Check whether bot is still valid
            if (!bot || !bot.entity) {
                throw new Error("Bot disconnected immediately after spawn");
            }

            // Shim bot.position → bot.entity.position.
            // LLMs often generate `bot.position.x` from pretraining
            // data (older mineflayer versions), but current mineflayer only exposes
            // `bot.entity.position`. Without this shim, `bot.position` is undefined
            // and `.x` throws "Cannot read properties of undefined (reading 'x')".
            if (!bot.position) {
                Object.defineProperty(bot, 'position', {
                    get: function() { return bot.entity?.position; },
                    configurable: true,
                });
            }

            // Apply night vision once on initial spawn (re-applied on
            // respawn via the respawn handler above).
            applyNightVision(bot);

            if (req.body.reset === "hard") {
                if (bot._posDump) bot._posDump("before-kill");
                bot.chat("/clear @s");
                bot.chat("/kill @s");
                if (bot._posDump) bot._posDump("after-kill-cmd");
                // Wait for respawn
                await bot.waitForTicks(bot.waitTicks || 5);
                if (bot._posDump) bot._posDump("after-kill-wait");
                
                // Recheck whether bot is still valid
                if (!bot || !bot.entity) {
                    throw new Error("Bot disconnected during hard reset");
                }
                
                const inventory = req.body.inventory ? req.body.inventory : {};
                const equipment = req.body.equipment
                    ? req.body.equipment
                    : [null, null, null, null, null, null];
                for (let key in inventory) {
                    bot.chat(`/give @s minecraft:${key} ${inventory[key]}`);
                    itemTicks += 1;
                }
                const equipmentNames = [
                    "armor.head",
                    "armor.chest",
                    "armor.legs",
                    "armor.feet",
                    "weapon.mainhand",
                    "weapon.offhand",
                ];
                for (let i = 0; i < 6; i++) {
                    if (i === 4) continue;
                    if (equipment[i]) {
                        bot.chat(
                            `/item replace entity @s ${equipmentNames[i]} with minecraft:${equipment[i]}`
                        );
                        itemTicks += 1;
                    }
                }
            }

            if (req.body.position) {
                // Check whether bot is still valid
                if (!bot || !bot.entity) {
                    throw new Error("Bot disconnected before position setup");
                }
                bot.chat(
                    `/tp @s ${req.body.position.x} ${req.body.position.y} ${req.body.position.z}`
                );
            }

            await completeSetup(itemTicks);
        } catch (err) {
            console.error("[Start] Error handling spawn event:", err);
            console.error("[Start] Error stack:", err.stack);
            if (!responseSent) {
                sendResponse({ 
                    error: err.message || String(err),
                    stack: err.stack 
                }, 500);
            }
        }
    });

    function onConnectionFailed(e) {
        console.error("[Start] Bot connection failed:", e);
        console.error("[Start] Connection-failure details:", e.message || String(e));
        if (e.stack) {
            console.error("[Start] Error stack:", e.stack);
        }
        if (bot) {
            try {
                bot.end();
            } catch (err) {
                console.error("[Start] Error while closing bot:", err);
            }
            bot = null;
        }
        sendResponse({ 
            error: e.message || String(e),
            code: e.code || "UNKNOWN",
            stack: e.stack 
        }, 400);
    }
    
    function onDisconnect(message) {
        console.log("[Start] Bot disconnected:", message);
        if (bot) {
            try {
                if (bot.viewer) {
                    bot.viewer.close();
                }
                bot.end();
            } catch (e) {
                console.error("[Start] Error while closing bot:", e);
            }
        }
        bot = null;
    }
});

app.post("/step", async (req, res) => {
    // PATCH (skillnet): a /step with no bot used to throw synchronously in an
    // async express-4 handler, which surfaces as an unhandled rejection.
    if (!bot) {
        res.status(500).json({ error: "Bot not started" });
        return;
    }
    // import useful package
    let response_sent = false;
    function otherError(err) {
        // PATCH (skillnet): this can fire from a stale listener after the bot
        // was torn down; dereferencing a null bot here would throw INSIDE an
        // uncaughtException handler, which is fatal and bypasses every
        // listener.
        if (!bot) {
            console.error("[Step] Uncaught error with no bot:", (err && err.stack) || err);
            if (!response_sent) {
                response_sent = true;
                try { res.status(500).json({ error: String((err && err.message) || err) }); } catch (e) {}
            }
            return;
        }
        console.log("Uncaught Error");
        bot.emit("error", handleError(err));
            bot.waitForTicks(bot.waitTicks).then(() => {
            if (!response_sent) {
                response_sent = true;
                try {
                    const obsResult = bot.observe();
                    safeJsonResponse(res, obsResult);
                } catch (obsErr) {
                    console.error(`[Step] Error in otherError handler: ${obsErr.message}`);
                    safeJsonResponse(res, {
                        error: 'Failed to observe bot state after error',
                        message: obsErr.message
                    }, 500);
                }
            }
        });
    }

    process.on("uncaughtException", otherError);

    // Abort the step early if the bot disconnects mid-execution. Without
    // this, user code's awaits on event-driven promises (pathfinder,
    // collectblock pickup, bot.dig blockUpdate) stay permanently pending
    // because the underlying events never fire after disconnect. The HTTP
    // /step request then never returns a response and the caller blocks
    // until the watcher's 30min cap kills the whole run. This abort path
    // short-circuits that failure: surface a 500 immediately so the
    // caller can advance to the next iteration. Cleanup in finally
    // detaches both listeners on normal completion.
    function onStepDisconnect(reason) {
        if (response_sent) return;
        const reasonStr = (reason && typeof reason === "object")
            ? (reason.message || JSON.stringify(reason))
            : String(reason);
        console.error(`[Step abort] bot disconnect during step: ${reasonStr}`);
        response_sent = true;
        if (!res.headersSent) {
            safeJsonResponse(res, {
                error: "Bot disconnected mid-step",
                reason: reasonStr,
            }, 500);
        }
    }
    bot.once("end", onStepDisconnect);
    bot.once("kicked", onStepDisconnect);

    // Wall-clock guardrail. The disconnect-abort above only fires when the
    // bot leaves the server. It does NOT cover a skill that loops forever
    // while the bot stays CONNECTED — e.g. ensureOakLogs trying to mine
    // straight up out of a deepslate pocket: y never increases, no exit
    // condition fires, and each iteration awaits pathfinder/physicsTick
    // events that keep firing, so the /step promise never resolves. Without
    // a server-side cap the caller blocks until its own read-timeout,
    // wasting many minutes per attempt. This timer bounds any such
    // step: surface a 500 so the caller advances, then bot.end() so the
    // wedged skill's event-driven awaits stop resolving and it cannot make
    // further progress. Budget is overridable for tests via env.
    // 600s: gather-heavy tasks (e.g. mining scattered obsidian across a
    // cratered field, then building and lighting a portal frame) were
    // observed making steady progress past the previous 240s budget, which
    // aborted legitimately-busy steps and discarded their events. Still
    // well under the client's request timeout so the server aborts and
    // replies before the caller gives up. Tunable via env.
    const STEP_WALLCLOCK_MS =
        (parseInt(process.env.PSN_STEP_WALLCLOCK_SEC, 10) || 600) * 1000;
    const stepWallclockTimer = setTimeout(() => {
        if (response_sent) return;
        console.error(
            `[Step abort] wall-clock budget ${STEP_WALLCLOCK_MS}ms exceeded; ` +
            `aborting step`
        );
        response_sent = true;
        if (!res.headersSent) {
            safeJsonResponse(res, {
                error: "Step exceeded wall-clock budget",
                budget_ms: STEP_WALLCLOCK_MS,
            }, 500);
        }
        // Neutralize the wedged skill: disconnect so its pending awaits on
        // pathfinder/dig/physicsTick events stop resolving. The next /step
        // (or the bridge's process restart) starts from a clean connection.
        try { bot.end("step-wallclock-timeout"); } catch (e) {}
    }, STEP_WALLCLOCK_MS);

    const mcData = require("minecraft-data")(bot.version);
    mcData.itemsByName["leather_cap"] = mcData.itemsByName["leather_helmet"];
    mcData.itemsByName["leather_tunic"] =
        mcData.itemsByName["leather_chestplate"];
    mcData.itemsByName["leather_pants"] =
        mcData.itemsByName["leather_leggings"];
    mcData.itemsByName["leather_boots"] = mcData.itemsByName["leather_boots"];
    mcData.itemsByName["lapis_lazuli_ore"] = mcData.itemsByName["lapis_ore"];
    mcData.blocksByName["lapis_lazuli_ore"] = mcData.blocksByName["lapis_ore"];
    const {
        Movements,
        goals: {
            Goal,
            GoalBlock,
            GoalNear,
            GoalXZ,
            GoalNearXZ,
            GoalY,
            GoalGetToBlock,
            GoalLookAtBlock,
            GoalBreakBlock,
            GoalCompositeAny,
            GoalCompositeAll,
            GoalInvert,
            GoalFollow,
            GoalPlaceBlock,
        },
        pathfinder,
        Move,
        ComputedPath,
        PartiallyComputedPath,
        XZCoordinates,
        XYZCoordinates,
        SafeBlock,
        GoalPlaceBlockOptions,
    } = require("mineflayer-pathfinder");
    const { Vec3 } = require("vec3");

    // Set up pathfinder
    const movements = new Movements(bot, mcData);
    // mineflayer-pathfinder hardcodes scafoldingBlocks (sic — typo in upstream library)
    // to only [dirt, cobblestone]. This means the pathfinder cannot tower-up using any
    // other placeable block. A bot deep underground may hold only stone-family blocks
    // (e.g. cobbled_deepslate) with no dirt or cobblestone, in which case upward
    // exploration fails because the pathfinder refuses to consider tower-up moves
    // (getMoveUp returns early when remainingBlocks==0). We extend the list with the
    // common stone-family blocks the bot is likely to accumulate while mining
    // underground, so the pathfinder can use whatever rocky inventory it has on hand.
    const extraScaffolding = [
        "cobbled_deepslate",
        "stone",
        "andesite",
        "diorite",
        "granite",
        "tuff",
    ];
    for (const name of extraScaffolding) {
        if (mcData.itemsByName[name]) {
            movements.scafoldingBlocks.push(mcData.itemsByName[name].id);
        }
    }
    bot.pathfinder.setMovements(movements);

    // Raise the pathfinder's A* planning budget from the library default (5s) to 10s.
    // Long/complex paths (e.g. escaping a deep mined-out pocket) can need more than 5s
    // of planning and otherwise fail with a thinkTimeout; 10s gives the planner room.
    bot.pathfinder.thinkTimeout = 10000; // ms (default 5000)

    // ============================================================
    // Enhanced recipe-check API - resolves recipesFor's inability to distinguish "recipe missing" from "materials missing"
    // Reuses material-analysis logic from craftHelper.js
    // ============================================================
    
    /**
     * Analyze recipe material requirements and find what is missing (reuses craftHelper.js algorithm)
     * @param {number} itemId - ID of the item to craft
     * @param {Block|null} craftingTable - crafting-table block; null for hand-crafting
     * @returns {{items: Array, message: string}} - list of missing materials and a formatted message
     */
    function analyzeRecipeMaterials(itemId, craftingTable) {
        const recipes = bot.recipesAll(itemId, null, craftingTable);
        if (!recipes.length) {
            return { items: [], message: 'no available recipes' };
        }

        // Compute the missing-material count for each recipe
        let minMissing = 999;
        const recipeMissingMap = [];

        for (const recipe of recipes) {
            let missing = 0;
            for (const delta of recipe.delta) {
                if (delta.count < 0) {
                    const invItem = bot.inventory.findInventoryItem(
                        mcData.items[delta.id].name, null
                    );
                    const have = invItem ? invItem.count : 0;
                    missing += Math.max(-delta.count - have, 0);
                }
            }
            recipeMissingMap.push({ recipe, missing });
            if (missing < minMissing) {
                minMissing = missing;
            }
        }

        // Collect all recipes with the fewest missing materials (tied recipes)
        const tiedRecipes = recipeMissingMap
            .filter(r => r.missing === minMissing)
            .map(r => r.recipe);

        if (!tiedRecipes.length) {
            return { items: [], message: 'no valid recipe found' };
        }

        const bestRecipe = tiedRecipes[0];

        // Generate the missing-materials list and detect variant groups
        const missingItems = [];
        for (const delta of bestRecipe.delta) {
            if (delta.count < 0) {
                const itemName = mcData.items[delta.id].name;
                const invItem = bot.inventory.findInventoryItem(itemName, null);
                const required = -delta.count;
                const have = invItem ? invItem.count : 0;
                if (have < required) {
                    const need = required - have;

                    // Detect variant groups: when multiple tied recipes share the same count but differ on item at this position
                    if (tiedRecipes.length > 1) {
                        const variantNames = _collectVariantNames(tiedRecipes, delta);
                        if (variantNames.length > 1) {
                            missingItems.push({
                                name: itemName,
                                need: need,
                                have: have,
                                required: required,
                                variants: variantNames,
                            });
                            continue;
                        }
                    }

                    missingItems.push({ name: itemName, need: need, have: have, required: required });
                }
            }
        }

        return {
            items: missingItems,
            message: missingItems.map(i => {
                if (i.variants) {
                    return `${i.need} planks (any single type: ${i.variants.join(', ')})`;
                }
                return `${i.need} ${i.name}`;
            }).join(', ')
        };
    }

    /**
     * Collect variant names at the same position across tied recipes
     * Used to detect recipe variant groups (e.g. oak_planks/birch_planks/spruce_planks)
     */
    function _collectVariantNames(tiedRecipes, referenceDelta) {
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
    
    /**
     * Enhanced recipe-check API
     * Unlike bot.recipesFor(), this API distinguishes "recipe missing" from "materials missing"
     * 
     * @param {string} itemName - name of the item to craft
     * @param {number} count - quantity to craft (default 1)
     * @param {Block|null} craftingTable - crafting-table block; null for hand-crafting
     * @returns {{exists: boolean, available: boolean, reason?: string, recipe?: Object, recipes?: Array, missing?: Array, message: string}}
     */
    bot.checkRecipe = function(itemName, count = 1, craftingTable = null) {
        const itemDef = mcData.itemsByName[itemName];
        if (!itemDef) {
            return { 
                exists: false, 
                available: false, 
                reason: 'unknown_item',
                message: `Unknown item: ${itemName}`
            };
        }
        
        // Check whether the recipe exists (ignoring materials)
        const allRecipes = mcData.recipes[itemDef.id];
        if (!allRecipes || allRecipes.length === 0) {
            return { 
                exists: false, 
                available: false, 
                reason: 'no_recipe',
                message: `No recipe exists for ${itemName} in this Minecraft version`
            };
        }
        
        // Check whether there is an executable recipe (considering materials)
        const available = bot.recipesFor(itemDef.id, null, count, craftingTable);
        if (available.length > 0) {
            return { 
                exists: true, 
                available: true, 
                recipe: available[0],
                recipes: available,
                message: `Can craft ${itemName}`
            };
        }
        
        // Detect the case where a 3x3 recipe lacks a crafting table
        // Critical: keep reason: 'insufficient_materials' because craftItem.js
        // checks checkResult.reason === 'insufficient_materials' to decide whether to call failedCraftFeedback.
        // Changing the reason would break craftItem (it would skip failedCraftFeedback → fail silently).
        if (!craftingTable) {
            // Two-step detection (same logic as craftHelper.js:failedCraftFeedback):
            // Step 1: check whether a no-table recipe exists
            const noTableRecipes = bot.recipesAll(itemDef.id, null, null);
            // Step 2: only when no 2x2 recipe exists, check for a 3x3 recipe
            if (!noTableRecipes || noTableRecipes.length === 0) {
                const tableRecipes = bot.recipesAll(itemDef.id, null, true);
                if (tableRecipes && tableRecipes.length > 0) {
                    return {
                        exists: true,
                        available: false,
                        reason: 'insufficient_materials',  // keep unchanged! see comment above
                        message: `Cannot craft ${itemName}: this is a 3x3 recipe that requires a crafting table. No crafting table block was provided or found nearby.`
                    };
                }
            }
            // If a 2x2 recipe exists but materials are insufficient, fall through to analyzeRecipeMaterials
        }

        // Recipe exists but materials are insufficient - analyze what is missing
        const missing = analyzeRecipeMaterials(itemDef.id, craftingTable);
        return {
            exists: true,
            available: false,
            reason: 'insufficient_materials',
            missing: missing.items,
            message: `Cannot craft ${itemName}: insufficient materials. Missing: ${missing.message}`
        };
    };
    
    console.log("[checkRecipe] Enhanced recipe-check API injected");
    // ============================================================

    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];

    function onTick() {
        bot.globalTickCounter++;
        if (bot.pathfinder.isMoving()) {
            bot.stuckTickCounter++;
            if (bot.stuckTickCounter >= 100) {
                onStuck(1.5);
                bot.stuckTickCounter = 0;
            }
        }
    }

    bot.on("physicsTick", onTick);

    // initialize fail count
    let _craftItemFailCount = 0;
    let _killMobFailCount = 0;
    let _mineBlockFailCount = 0;
    let _mineBlockZeroGainCount = 0;
    let _placeItemFailCount = 0;
    let _smeltItemFailCount = 0;

    // Retrieve array form post bod
    const code = req.body.code;
    const programs = req.body.programs;
    const skillNames = req.body.skill_names || []; // new: list of skill names
    // keep_paused: world-setup commands run while the server is paused, so the
    // tick-waits below MUST be skipped (waitForTicks would hang with no ticks).
    const keepPaused = req.body.keep_paused === true;
    bot.cumulativeObs = [];

    // Initialize skill-execution event storage
    bot.skillExecutions = [];

    try {
        // PATCH (skillnet): epoch-stamped step ownership. A wallclock-aborted
        // step never reaches its finally, and the client retries /step on
        // timeouts, so a stale finally can fire while a NEWER step is live;
        // the epoch gate keeps it from touching the live step. The entry
        // quarantine is the backstop that actually protects attribution:
        // whatever the previous step leaked (goal, collect task, dig, pvp,
        // controls) is cleared before this step's code runs. Bridge errors
        // queued between steps are flushed AFTER the cumulativeObs reset
        // above, so they surface in THIS step's observations.
        var myEpoch = ++stepEpoch;
        stepInFlight = true;
        quarantineActuators();
        while (pendingBridgeErrors.length) {
            try { bot.emit("error", new Error(pendingBridgeErrors.shift())); } catch (e) {}
        }
        if (!keepPaused) {
            console.log(`[Step] Starting step execution, waiting ${bot.waitTicks} ticks...`);
            try {
                await bot.waitForTicks(bot.waitTicks);
                console.log(`[Step] Ticks waited, evaluating code...`);
            } catch (err) {
                console.log(`[Step] Warning: waitForTicks failed: ${err.message}, continuing anyway...`);
            }
        }
        const r = await evaluateCode(code, programs, skillNames, mcData, Vec3);
        console.log(`[Step] Code evaluation result: ${r}`);
        process.off("uncaughtException", otherError);
        if (r !== "success") {
            bot.emit("error", handleError(r));
        }
        await returnItems();
        // wait for last message
        console.log(`[Step] Waiting ${bot.waitTicks} ticks after code execution...`);
        // Bound ONLY this post-execution settle-wait (a short "let the last
        // messages flush" wait) — NOT in-skill waitForTicks, which must run
        // full length (e.g. smelting does waitForTicks(12*20)=12s). After a
        // skill error physicsTick can stall, leaving mineflayer's native
        // waitForTicks here to hang until the 240s wall-clock guardrail, which
        // returns a generic 500 that the client turns into a hard error —
        // bypassing the normal error-feedback path. The event loop stays free
        // during the stall, so racing a short settle timeout lets /step deliver
        // the already-captured onError observation (200) promptly, so the
        // error is surfaced through the normal path. In-skill waits
        // are left native/uncapped, so legitimate long waits (smelting) work.
        const STEP_SETTLE_MS = parseInt(process.env.PSN_STEP_SETTLE_MS, 10) || 5000;
        if (!keepPaused) {
            try {
                await Promise.race([
                    bot.waitForTicks(bot.waitTicks),
                    new Promise((resolve) => setTimeout(resolve, STEP_SETTLE_MS)),
                ]);
                console.log(`[Step] Final ticks waited (or settle timeout), preparing response...`);
            } catch (err) {
                console.log(`[Step] Warning: final waitForTicks failed: ${err.message}, continuing anyway...`);
            }
        }
        
        // Append skill-execution events to cumulativeObs
        if (bot.skillExecutions && bot.skillExecutions.length > 0) {
            for (const exec of bot.skillExecutions) {
                bot.cumulativeObs.push([exec.type, exec]);
            }
            bot.skillExecutions = []; // clear to avoid duplicates
        }

        // Inject placed_blocks evidence from returnItems()
        if (bot.placedBlocksThisStep && bot.placedBlocksThisStep.length > 0) {
            bot.cumulativeObs.push(["placedBlocks", {
                placedBlocks: bot.placedBlocksThisStep,
            }]);
        }

        if (!response_sent) {
            response_sent = true;
            try {
                const obsResult = bot.observe();
                safeJsonResponse(res, obsResult);
            } catch (obsErr) {
                console.error(`[Step] Error in bot.observe(): ${obsErr.message}`);
                if (obsErr.stack) {
                    console.error(`[Step] Stack: ${obsErr.stack}`);
                }
                // Send error response
                safeJsonResponse(res, {
                    error: 'Failed to observe bot state',
                    message: obsErr.message
                }, 500);
            }
        }
    } catch (err) {
        console.error(`[Step] Error during step execution: ${err.message}`);
        if (err.stack) {
            console.error(`[Step] Stack: ${err.stack}`);
        }
        // Ensure a response is sent even on error
        if (!response_sent) {
            response_sent = true;
            try {
                const obsResult = bot.observe();
                safeJsonResponse(res, obsResult);
            } catch (responseErr) {
                console.error(`[Step] Failed to send error response: ${responseErr.message}`);
                if (responseErr.stack) {
                    console.error(`[Step] Response error stack: ${responseErr.stack}`);
                }
                // If JSON cannot be sent, try sending an error status
                if (!res.headersSent) {
                    safeJsonResponse(res, {
                        error: err.message || 'Unknown error',
                        originalError: responseErr.message
                    }, 500);
                }
            }
        }
    } finally {
        clearTimeout(stepWallclockTimer);
        // PATCH (skillnet): the bot can be nulled mid-step (kick/disconnect);
        // a throw here would itself become an unhandled rejection.
        if (bot) bot.removeListener("physicsTick", onTick);
        process.off("uncaughtException", otherError);
        // Detach the disconnect-abort listeners if the step completed
        // normally; bot.once already removes itself on fire so this is a
        // no-op when the abort path actually ran.
        if (bot) {
            bot.removeListener("end", onStepDisconnect);
            bot.removeListener("kicked", onStepDisconnect);
        }
        // PATCH (skillnet): only the CURRENT step may clear the flag and
        // quarantine; a stale finally from an abandoned step must not cancel
        // the live step's actuators.
        if (myEpoch === stepEpoch) {
            stepInFlight = false;
            quarantineActuators();
        }
    }

    async function evaluateCode(code, programs, skillNames = [], mcData, Vec3) {
        // Isolate each skill's module-top declarations behind an IIFE
        // and re-export top-level fn declarations to globalThis. This prevents
        // LLM-generated `let mcData = null;` at module top from colliding with
        // globalDepsCode's `var mcData` or across skill files. No-op for skills
        // that are already well-formed.
        programs = wrapProgramsInIIFE(programs);

        // Ensure mcData and Vec3 are available in the global scope (used by skill code and helper functions)
        // These variables are declared in the /step handler scope and must be injected into the execution context via eval
        // Use the global object to set globals so they are accessible in all executed code
        // Note: do not use `if (typeof Vec3 === 'undefined')` here to check Vec3
        // Because if user code contains `const { Vec3 } = require("vec3");`,
        // JavaScript's Temporal Dead Zone (TDZ) rules cause an error when Vec3 is accessed before declaration:
        // "Cannot access 'Vec3' before initialization"
        // 
        // Solution: always access Vec3 via global.Vec3 to avoid TDZ issues
        const globalDepsCode = `
            // Inject global dependencies so skill code and helper functions can access them
            if (typeof global.mcData === 'undefined') {
                global.mcData = require('minecraft-data')(bot.version);
                global.mcData.itemsByName["leather_cap"] = global.mcData.itemsByName["leather_helmet"];
                global.mcData.itemsByName["leather_tunic"] = global.mcData.itemsByName["leather_chestplate"];
                global.mcData.itemsByName["leather_pants"] = global.mcData.itemsByName["leather_leggings"];
                global.mcData.itemsByName["leather_boots"] = global.mcData.itemsByName["leather_boots"];
                global.mcData.itemsByName["lapis_lazuli_ore"] = global.mcData.itemsByName["lapis_ore"];
                global.mcData.blocksByName["lapis_lazuli_ore"] = global.mcData.blocksByName["lapis_ore"];
            }
            // Ensure Vec3 is always available in the global scope
            // Use global.Vec3 here to avoid conflicting with 'const Vec3' declarations in user code
            if (typeof global.Vec3 === 'undefined') {
                global.Vec3 = require('vec3').Vec3;
            }
            // Create local-variable references for ergonomic use (use var, not const/let, to avoid TDZ)
            // var is hoisted to the top of the function and produces no TDZ
            if (typeof mcData === 'undefined') {
                var mcData = global.mcData;
            }
            // Declare Vec3 with var so it is accessible anywhere without triggering TDZ
            // If user code uses const { Vec3 } = require("vec3"), the user declaration shadows this
            // But since var produces no TDZ, the checks and assignments here will not error
            var Vec3 = global.Vec3;
            
            // Expose bot.checkRecipe for direct use by skill code
            // checkRecipe is the enhanced recipe-check API that distinguishes "recipe missing" from "materials missing"
            if (typeof global.checkRecipe === 'undefined' && bot.checkRecipe) {
                global.checkRecipe = bot.checkRecipe.bind(bot);
            }
            var checkRecipe = global.checkRecipe;
        `;
        
        // If a skill-name list is provided, create wrappers
        if (skillNames && skillNames.length > 0) {
            console.log(`[Skill Wrapper] Received ${skillNames.length} skill names for wrapping`);
            
            // Build wrapper code for each skill
            // Note: callStack must be defined in fullCode because the wrapper code executes as a string
            const wrapperCode = skillNames.map((skillName, index) => {
                // Escape skillName to prevent code injection
                const safeSkillName = skillName.replace(/[^a-zA-Z0-9_]/g, '_');
                return `
                (function() {
                    try {
                        // Check whether the function exists (supports multiple definition styles)
                        if (typeof ${skillName} === 'undefined' || typeof ${skillName} !== 'function') {
                            // Function does not exist; skip wrapping
                            console.log('[Skill Wrapper] Skipping ${safeSkillName}: function not found or not a function');
                            return;
                        }
                        
                        // Check whether the function is already wrapped (by checking for a marker in the function name)
                        // If already wrapped, skip re-wrapping
                        const funcStr = ${skillName}.toString();
                        if (funcStr.includes('skillStart') || funcStr.includes('skillExecutions')) {
                            // Function already wrapped; skip
                            console.log('[Skill Wrapper] Skipping ${safeSkillName}: already wrapped');
                            return;
                        }
                        
                        // Save reference to the original function
                        const originalFunc = ${skillName};
                        console.log('[Skill Wrapper] Wrapping ${safeSkillName}...');

                        // ========== Detect whether the function is async ==========
                        const isAsyncFunction = originalFunc.constructor.name === 'AsyncFunction';
                        console.log('[Skill Wrapper] ${safeSkillName} isAsync:', isAsyncFunction);

                        // ========== Helper that auto-injects the bot parameter ==========
                        function autoInjectBot(args) {
                            const firstArgIsBot = args.length > 0 &&
                                                  args[0] &&
                                                  typeof args[0] === 'object' &&
                                                  args[0].entity;
                            if (!firstArgIsBot) {
                                console.log('[Skill Wrapper] Auto-injecting bot for ${safeSkillName}');
                                return [bot, ...args];
                            }
                            return args;
                        }

                        // ========== Choose wrapping strategy based on function type ==========
                        if (!isAsyncFunction) {
                            // ===== SYNC function wrapping: synchronously record skillStart + skillEnd =====
                            const syncOriginal = originalFunc;
                            ${skillName} = function(...args) {
                                // Auto-inject the bot parameter
                                args = autoInjectBot(args);

                                const startTime = Date.now();
                                __skillCallStack__.push('${safeSkillName}');

                                // Synchronously record skillStart (before execution, callStack intact)
                                let serializedArgs = null;
                                if (args.length > 0) {
                                    try {
                                        // Filter out the bot object (usually the first argument)
                                        const safeArgs = args.map(arg => {
                                            if (arg && typeof arg === 'object' && arg.entity) {
                                                return '[bot]';  // replace the bot object
                                            }
                                            if (typeof arg === 'bigint') {
                                                return arg.toString();
                                            }
                                            return arg;
                                        });
                                        serializedArgs = JSON.stringify(safeArgs.slice(1));  // exclude bot parameter
                                    } catch (e) {
                                        serializedArgs = "[serialization error]";
                                    }
                                }

                                // Synchronously record skillStart (with a callStack snapshot)
                                try {
                                    if (bot && bot.skillExecutions) {
                                        bot.skillExecutions.push({
                                            type: "skillStart",
                                            skillName: "${safeSkillName}",
                                            timestamp: startTime,
                                            callStack: __skillCallStack__.slice(),  // callStack snapshot
                                            args: serializedArgs,
                                            isSync: true
                                        });
                                    }
                                } catch (e) { /* ignore tracking errors */ }

                                try {
                                    const result = syncOriginal.apply(this, args);

                                    // Synchronously record skillEnd (no setTimeout)
                                    try {
                                        if (bot && bot.skillExecutions) {
                                            bot.skillExecutions.push({
                                                type: "skillEnd",
                                                skillName: "${safeSkillName}",
                                                timestamp: Date.now(),
                                                duration: Date.now() - startTime,
                                                success: true,
                                                isSync: true
                                            });
                                        }
                                    } catch (e) { /* ignore tracking errors */ }

                                    return result;  // return the synchronous result immediately
                                } catch (err) {
                                    // Synchronously record skillError (no setTimeout)
                                    try {
                                        if (bot && bot.skillExecutions) {
                                            bot.skillExecutions.push({
                                                type: "skillError",
                                                skillName: "${safeSkillName}",
                                                timestamp: Date.now(),
                                                duration: Date.now() - startTime,
                                                success: false,
                                                error: err.message || String(err),
                                                stack: err.stack || null,
                                                isSync: true
                                            });
                                        }
                                    } catch (e) { /* ignore tracking errors */ }
                                    throw err;
                                } finally {
                                    __skillCallStack__.pop();
                                }
                            };
                        } else {
                            // ===== ASYNC function wrapping: original logic =====
                            ${skillName} = async function(...args) {
                                // Auto-inject the bot parameter
                                args = autoInjectBot(args);

                                // Diagnostic: confirm the wrapped function was called
                                console.log('[WRAPPED ${safeSkillName}] Function called');
                                console.log('[WRAPPED ${safeSkillName}] bot exists:', !!bot);
                                console.log('[WRAPPED ${safeSkillName}] bot.skillExecutions exists:', !!(bot && bot.skillExecutions));

                                const startTime = Date.now();
                                __skillCallStack__.push('${safeSkillName}');
                            
                            // Safely serialize arguments (filter out the bot object to avoid circular-reference errors)
                            let serializedArgs = null;
                            if (args.length > 0) {
                                try {
                                    // Filter out the bot object (usually the first argument)
                                    const safeArgs = args.map(arg => {
                                        if (arg && typeof arg === 'object' && arg.entity) {
                                            return '[bot]';  // replace the bot object
                                        }
                                        if (typeof arg === 'bigint') {
                                            return arg.toString();
                                        }
                                        return arg;
                                    });
                                    serializedArgs = JSON.stringify(safeArgs, (key, value) => {
                                        if (typeof value === 'bigint') {
                                            return value.toString();
                                        }
                                        // Handle other potentially-circular objects
                                        if (value && typeof value === 'object' && value.entity) {
                                            return '[bot]';
                                        }
                                        return value;
                                    });
                                } catch (e) {
                                    serializedArgs = '[serialization failed]';
                                }
                            }
                            
                            // Safely capture pre-execution state (guarded by try/catch)
                            let preState = null;
                            try {
                                if (bot && bot.entity && bot.inventory) {
                                    preState = {
                                        inventory: {},
                                        position: {x: bot.entity.position.x, y: bot.entity.position.y, z: bot.entity.position.z},
                                        dimension: bot.game ? bot.game.dimension : undefined,
                                        equipment: {
                                            head: bot.inventory.slots[5] ? bot.inventory.slots[5].name : null,
                                            torso: bot.inventory.slots[6] ? bot.inventory.slots[6].name : null,
                                            legs: bot.inventory.slots[7] ? bot.inventory.slots[7].name : null,
                                            feet: bot.inventory.slots[8] ? bot.inventory.slots[8].name : null,
                                            hand: bot.inventory.slots[36] ? bot.inventory.slots[36].name : null,
                                            "off-hand": bot.inventory.slots[45] ? bot.inventory.slots[45].name : null
                                        }
                                    };
                                    const items = bot.inventory.items();
                                    for (const item of items) {
                                        preState.inventory[item.name] = (preState.inventory[item.name] || 0) + item.count;
                                    }
                                }
                            } catch (stateErr) {
                                // State capture failed; continue (does not affect main logic)
                                preState = null;
                            }
                            
                            // Record skill execution start
                            if (bot.skillExecutions) {
                                bot.skillExecutions.push({
                                    type: "skillStart",
                                    skillName: "${safeSkillName}",
                                    timestamp: startTime,
                                    callStack: __skillCallStack__.slice(),
                                    args: serializedArgs,
                                    state: preState
                                });
                            }
                            
                            try {
                                const result = await originalFunc.apply(this, args);
                                const endTime = Date.now();
                                __skillCallStack__.pop();
                                
                                // Safely capture post-execution state (guarded by try/catch)
                                let postState = null;
                                try {
                                    if (bot && bot.entity && bot.inventory) {
                                        postState = {
                                            inventory: {},
                                            position: {x: bot.entity.position.x, y: bot.entity.position.y, z: bot.entity.position.z},
                                            dimension: bot.game ? bot.game.dimension : undefined,
                                            equipment: {
                                                head: bot.inventory.slots[5] ? bot.inventory.slots[5].name : null,
                                                torso: bot.inventory.slots[6] ? bot.inventory.slots[6].name : null,
                                                legs: bot.inventory.slots[7] ? bot.inventory.slots[7].name : null,
                                                feet: bot.inventory.slots[8] ? bot.inventory.slots[8].name : null,
                                                hand: bot.inventory.slots[36] ? bot.inventory.slots[36].name : null,
                                                "off-hand": bot.inventory.slots[45] ? bot.inventory.slots[45].name : null
                                            }
                                        };
                                        const items = bot.inventory.items();
                                        for (const item of items) {
                                            postState.inventory[item.name] = (postState.inventory[item.name] || 0) + item.count;
                                        }
                                    }
                                } catch (stateErr) {
                                    // State capture failed; continue
                                    postState = null;
                                }
                                
                                // Record skill execution success
                                if (bot.skillExecutions) {
                                    bot.skillExecutions.push({
                                        type: "skillEnd",
                                        skillName: "${safeSkillName}",
                                        timestamp: endTime,
                                        duration: endTime - startTime,
                                        success: true,
                                        state: postState
                                    });
                                }
                                
                                return result;
                            } catch (err) {
                                const endTime = Date.now();
                                __skillCallStack__.pop();
                                
                                // Safely capture post-execution state (state may change even on failure)
                                let postState = null;
                                try {
                                    if (bot && bot.entity && bot.inventory) {
                                        postState = {
                                            inventory: {},
                                            position: {x: bot.entity.position.x, y: bot.entity.position.y, z: bot.entity.position.z},
                                            dimension: bot.game ? bot.game.dimension : undefined,
                                            equipment: {
                                                head: bot.inventory.slots[5] ? bot.inventory.slots[5].name : null,
                                                torso: bot.inventory.slots[6] ? bot.inventory.slots[6].name : null,
                                                legs: bot.inventory.slots[7] ? bot.inventory.slots[7].name : null,
                                                feet: bot.inventory.slots[8] ? bot.inventory.slots[8].name : null,
                                                hand: bot.inventory.slots[36] ? bot.inventory.slots[36].name : null,
                                                "off-hand": bot.inventory.slots[45] ? bot.inventory.slots[45].name : null
                                            }
                                        };
                                        const items = bot.inventory.items();
                                        for (const item of items) {
                                            postState.inventory[item.name] = (postState.inventory[item.name] || 0) + item.count;
                                        }
                                    }
                                } catch (stateErr) {
                                    // State capture failed; continue
                                    postState = null;
                                }

                                // ======== New: enrich error messages with diagnostic facts ========
                                if (err.message && postState && postState.position) {
                                    // Detect blockUpdate timeout and attach diagnostic facts
                                    if (err.message.includes("blockUpdate") && (err.message.includes("timeout") || err.message.includes("did not fire"))) {
                                        // Updated regex: supports negative coordinates and format variants
                                        const blockUpdateMatch = err.message.match(/blockUpdate[:\\s]*\\(?(-?\\d+),\\s*(-?\\d+),\\s*(-?\\d+)\\)?/);
                                        if (blockUpdateMatch) {
                                            const targetPos = {
                                                x: parseInt(blockUpdateMatch[1]),
                                                y: parseInt(blockUpdateMatch[2]),
                                                z: parseInt(blockUpdateMatch[3])
                                            };
                                            const botPos = {
                                                x: Math.floor(postState.position.x),
                                                y: Math.floor(postState.position.y),
                                                z: Math.floor(postState.position.z)
                                            };

                                            // Collect pure-fact diagnostics (no hints or solutions)
                                            const diagnosticFacts = {
                                                bot_position: botPos,
                                                target_position: targetPos,
                                                positions_equal: botPos.x === targetPos.x && botPos.y === targetPos.y && botPos.z === targetPos.z,
                                                distance: Math.sqrt(
                                                    Math.pow(botPos.x - targetPos.x, 2) +
                                                    Math.pow(botPos.y - targetPos.y, 2) +
                                                    Math.pow(botPos.z - targetPos.z, 2)
                                                ).toFixed(1)
                                            };

                                            // Check whether the target position is occupied
                                            try {
                                                const targetBlock = bot.blockAt(new Vec3(targetPos.x, targetPos.y, targetPos.z));
                                                diagnosticFacts.block_at_target = targetBlock?.name || "air";
                                                diagnosticFacts.target_occupied = targetBlock?.name !== "air";
                                            } catch (blockErr) { /* ignore */ }

                                            // Attach facts to the error message (LLM can parse and reason)
                                            err.message = err.message + "\\n[DIAGNOSTIC_FACTS] " + JSON.stringify(diagnosticFacts);
                                            console.log(\`[DiagnosticEnhancement] Added facts to blockUpdate error: positions_equal=\${diagnosticFacts.positions_equal}\`);
                                        }
                                    }
                                }
                                // ======== End of enriched error message ========

                                // Record skill execution failure
                                if (bot.skillExecutions) {
                                    bot.skillExecutions.push({
                                        type: "skillError",
                                        skillName: "${safeSkillName}",
                                        timestamp: endTime,
                                        duration: endTime - startTime,
                                        success: false,
                                        error: err.message || String(err),
                                        stack: err.stack || null,
                                        state: postState
                                    });
                                }
                                
                                throw err; // rethrow the error
                            }
                        };
                        } // ===== end else (async function wrapping) =====

                        // Diagnostic: confirm wrapping completed
                        console.log('[Skill Wrapper] ${safeSkillName} wrapped successfully');
                        console.log('[Skill Wrapper] ${safeSkillName}.toString().slice(0,80):', ${skillName}.toString().slice(0, 80));
                    } catch (e) {
                        // If wrapping fails, log a warning but continue execution
                        console.log(\`Warning: Could not wrap skill ${safeSkillName}: \${e.message}\`);
                        if (e.stack) {
                            console.log(\`Stack: \${e.stack}\`);
                        }
                    }
                })();
                `;
            }).join('\n');
            
            // Execution order:
            // 1. Execute programs in the global scope to define all skill functions
            // 2. Execute the first wrapper in the global scope to wrap functions defined in programs
            // 3. Inside an async function, execute the second wrapper first (uses JS function-declaration hoisting)
            // 4. Then execute code (calls already see the wrapped versions)
            // 
            // Key: JavaScript function declarations are hoisted to the top of their scope
            // So "async function foo() {...}" already exists at the start of the async IIFE
            // We wrap before code runs so the call site sees the wrapped function
            // Sentinel markers let handleError compute the true line numbers
            // of `programs` and `code` inside the eval'd anonymous source.
            // Without them, the stack frame line (e.g. <anonymous>:795:35) is
            // offset by the size of globalDepsCode + wrappers, so reporting
            // the source line at that index points at the wrong code (or runs
            // off the end → "(line not available)" in the error feedback).
            const PROGRAMS_MARKER = "// __PSN_PROGRAMS_START__";
            const CODE_MARKER = "// __PSN_CODE_START__";
            const fullCode = `
                // First inject the global dependencies
                ${globalDepsCode}

                // Create the call-stack tracker (must be defined before wrapper code)
                // This variable lives in the global scope for wrapper code to use
                const __skillCallStack__ = [];

                ${PROGRAMS_MARKER}
                ${programs}
                // First wrapping pass: wrap functions defined in programs (in the global scope)
                ${wrapperCode}
                // Execute user code (in an async function so await is supported)
                (async () => {
                    // Second wrapping pass: wrap functions newly defined in code
                    // Function-declaration hoisting means the function already exists and can be wrapped
                    // Wrapping must finish before code actually executes (is called)
                    ${wrapperCode}
                    ${CODE_MARKER}
                    ${code}
                })()
            `;

            // Compute the 1-indexed eval-source line where programs / code start.
            // findIndex returns 0-indexed; +1 for 1-indexed line, +1 to skip past marker.
            const fullCodeLines = fullCode.split('\n');
            const _findMarker = (m) => fullCodeLines.findIndex(l => l.trim() === m);
            const programsStartLine = _findMarker(PROGRAMS_MARKER) + 2;
            const codeStartLine = _findMarker(CODE_MARKER) + 2;
            const programsLineCount = programs.split('\n').length;
            const codeLineCount = code.split('\n').length;
            const evalLineOffsets = {
                programsStart: programsStartLine,
                programsEnd: programsStartLine + programsLineCount - 1,
                codeStart: codeStartLine,
                codeEnd: codeStartLine + codeLineCount - 1,
            };
            
            // Diagnostic log: helps debug "Invalid or unexpected token" errors
            console.log('[DEBUG] === fullCode Stats ===');
            console.log('[DEBUG] fullCode length:', fullCode.length);
            console.log('[DEBUG] programs length:', programs.length);
            console.log('[DEBUG] code length:', code.length);
            console.log('[DEBUG] wrapperCode length:', wrapperCode.length);
            console.log('[DEBUG] skillNames:', JSON.stringify(skillNames));

            // Write fullCode to a temp file for analysis
            const fs = require('fs');
            fs.writeFileSync('/tmp/debug_fullCode.js', fullCode);
            console.log('[DEBUG] fullCode saved to /tmp/debug_fullCode.js');

            try {
                await eval(fullCode);
                return "success";
            } catch (err) {
                console.error('[DEBUG] eval FAILED:', err.message);
                if (err instanceof SyntaxError) {
                    // Try Node.js's vm module to get more detailed error info
                    try {
                        const vm = require('vm');
                        vm.compileFunction(fullCode);
                    } catch (vmErr) {
                        console.error('[DEBUG] vm.compileFunction error:', vmErr.message);
                    }
                }
                // Attach eval-line offsets so handleError can map a stack frame
                // like <anonymous>:795:35 to the actual line in `programs` or `code`.
                err._psnEvalOffsets = evalLineOffsets;
                return err;
            }
        } else {
            // If no skill-name list is given, use the original execution path
            try {
                await eval(`${globalDepsCode}\n(async () => {${programs}\n${code}})()`);
                return "success";
            } catch (err) {
                return err;
            }
        }
    }

    function onStuck(posThreshold) {
        // Defensive check - bot.entity may be undefined on death/disconnect
        if (!bot.entity) {
            console.warn("[onStuck] bot.entity is undefined, skipping stuck check");
            return;
        }
        const currentPos = bot.entity.position;
        bot.stuckPosList.push(currentPos);

        // Check if the list is full
        if (bot.stuckPosList.length === 5) {
            const oldestPos = bot.stuckPosList[0];
            const posDifference = currentPos.distanceTo(oldestPos);

            if (posDifference < posThreshold) {
                teleportBot(); // execute the function
            }

            // Remove the oldest time from the list
            bot.stuckPosList.shift();
        }
    }

    function teleportBot() {
        const blocks = bot.findBlocks({
            matching: (block) => {
                return block.type === 0;
            },
            maxDistance: 1,
            count: 27,
        });

        if (blocks) {
            // console.log(blocks.length);
            const randomIndex = Math.floor(Math.random() * blocks.length);
            const block = blocks[randomIndex];
            bot.chat(`/tp @s ${block.x} ${block.y} ${block.z}`);
        } else {
            bot.chat("/tp @s ~ ~1.25 ~");
        }
    }

    function returnItems() {
        // Record placed blocks before destroying them.
        // returnItems() is the authoritative source for placed_blocks evidence,
        // used to evaluate block-placement tasks correctly.
        const placedBlocks = [];
        bot.chat("/gamerule doTileDrops false");
        const crafting_table = bot.findBlock({
            matching: mcData.blocksByName.crafting_table.id,
            maxDistance: 128,
        });
        if (crafting_table) {
            placedBlocks.push("crafting_table");
            bot.chat(
                `/setblock ${crafting_table.position.x} ${crafting_table.position.y} ${crafting_table.position.z} air destroy`
            );
            bot.chat("/give @s crafting_table");
        }
        const furnace = bot.findBlock({
            matching: mcData.blocksByName.furnace.id,
            maxDistance: 128,
        });
        if (furnace) {
            placedBlocks.push("furnace");
            bot.chat(
                `/setblock ${furnace.position.x} ${furnace.position.y} ${furnace.position.z} air destroy`
            );
            bot.chat("/give @s furnace");
        }
        if (bot.inventoryUsed() >= 32) {
            // if chest is not in bot's inventory
            if (!bot.inventory.items().find((item) => item.name === "chest")) {
                bot.chat("/give @s chest");
            }
        }
        // if iron_pickaxe not in bot's inventory and bot.iron_pickaxe
        if (
            bot.iron_pickaxe &&
            !bot.inventory.items().find((item) => item.name === "iron_pickaxe")
        ) {
            bot.chat("/give @s iron_pickaxe");
        }
        bot.chat("/gamerule doTileDrops true");
        // Store placed blocks for observation
        bot.placedBlocksThisStep = placedBlocks;
    }

    function handleError(err) {
        let stack = err.stack;
        if (!stack) {
            return err;
        }
        console.log(stack);
        const final_line = stack.split("\n")[1];
        const regex = /<anonymous>:(\d+):\d+\)/;

        // Pre-split sources for bounds checks
        const codeLines = code.split("\n");
        const programLines = programs.split("\n");

        // Helper for safely fetching source lines (1-indexed)
        function safeGetLine(lines, index) {
            if (index > 0 && index <= lines.length) {
                return lines[index - 1].trim();
            }
            return null;
        }

        // Pull the eval-line offsets attached by evaluateCode. Without them
        // we fall back to a heuristic that uses programs.length as offset
        // (which under-counts by the size of globalDepsCode and was the
        // long-standing source of "(line not available)" in the error feedback).
        const offsets = err._psnEvalOffsets || null;
        const programs_length_fallback = programs.split("\n").length;

        // Find first stack frame inside the eval'd anonymous source AND
        // translate it to (section, source-line). Section is 'programs',
        // 'code', or null (frame is inside wrapper/prefix and not directly
        // attributable to user-visible source).
        let firstFrameInfo = null;  // { section, sourceLine, evalLine }
        let userCodeLine = null;    // line within `code` for any frame that falls there
        for (const line of stack.split("\n")) {
            const match = regex.exec(line);
            if (!match) continue;
            const evalLine = parseInt(match[1]);

            if (offsets) {
                if (evalLine >= offsets.programsStart && evalLine <= offsets.programsEnd) {
                    const srcLine = evalLine - offsets.programsStart + 1;
                    if (!firstFrameInfo) firstFrameInfo = { section: "programs", sourceLine: srcLine, evalLine };
                } else if (evalLine >= offsets.codeStart && evalLine <= offsets.codeEnd) {
                    const srcLine = evalLine - offsets.codeStart + 1;
                    if (!firstFrameInfo) firstFrameInfo = { section: "code", sourceLine: srcLine, evalLine };
                    if (userCodeLine === null) userCodeLine = srcLine;
                }
            } else {
                // Legacy fallback: assume programs start at line 1 (wrong but
                // preserved so non-eval callers behave as before).
                if (evalLine >= programs_length_fallback && userCodeLine === null) {
                    userCodeLine = evalLine - programs_length_fallback;
                }
                if (!firstFrameInfo) firstFrameInfo = { section: null, sourceLine: null, evalLine };
            }
        }

        if (!firstFrameInfo) {
            return err.message;
        }

        let f_line = final_line.match(
            /\((?<file>.*):(?<line>\d+):(?<pos>\d+)\)/
        );
        if (f_line && f_line.groups && fs.existsSync(f_line.groups.file)) {
            const { file, line } = f_line.groups;
            const f = fs.readFileSync(file, "utf8").split("\n");
            const fileLine = safeGetLine(f, parseInt(line));
            let source = file + `:${line}\n${fileLine || "(line not available)"}\n `;

            const codeLine = userCodeLine !== null ? safeGetLine(codeLines, userCodeLine) : null;
            const code_source = codeLine
                ? "at " + codeLine + " in your code"
                : "";
            return source + err.message + (code_source ? "\n" + code_source : "");
        } else if (
            f_line &&
            f_line.groups &&
            f_line.groups.file.includes("<anonymous>")
        ) {
            // Use the first-frame info (now correctly offset by globalDepsCode +
            // wrappers) to fetch the actual source line that threw.
            let source;
            if (firstFrameInfo.section === "programs") {
                const progLine = safeGetLine(programLines, firstFrameInfo.sourceLine);
                source = `In your program code: ${progLine || "(line not available)"} (programs line ${firstFrameInfo.sourceLine})\n`;
            } else if (firstFrameInfo.section === "code") {
                const codeLine = safeGetLine(codeLines, firstFrameInfo.sourceLine);
                source = `In your code: ${codeLine || "(line not available)"} (code line ${firstFrameInfo.sourceLine})\n`;
            } else {
                // No section attribution — fall back to legacy "Your code:N"
                source = `Your code:${firstFrameInfo.evalLine}\n(line not available — frame in prefix or wrapper)\n`;
            }
            const userLine = userCodeLine !== null ? safeGetLine(codeLines, userCodeLine) : null;
            const code_source = userLine
                ? `at line ${userCodeLine}:${userLine} in your code`
                : "";
            return source + err.message + (code_source ? "\n" + code_source : "");
        }
        return err.message;
    }
});

// Safe JSON stringify function to prevent V8 fatal errors
function safeStringify(obj, space = null) {
    const seen = new WeakSet();
    const replacer = (key, value) => {
        // Handle circular references
        if (typeof value === 'object' && value !== null) {
            if (seen.has(value)) {
                return '[Circular]';
            }
            seen.add(value);
        }
        
        // Handle non-serializable values
        if (typeof value === 'function') {
            return '[Function]';
        }
        if (typeof value === 'undefined') {
            return null;
        }
        if (typeof value === 'symbol') {
            return value.toString();
        }
        if (typeof value === 'bigint') {
            return value.toString();
        }
        
        // Remove bot object references (avoid circular references)
        if (value && value.constructor && value.constructor.name === 'Bot') {
            return '[Bot Object]';
        }
        
        // Remove event emitter references
        if (value && typeof value.on === 'function' && typeof value.emit === 'function') {
            return '[EventEmitter]';
        }
        
        return value;
    };
    
    try {
        return JSON.stringify(obj, replacer, space);
    } catch (err) {
        console.error('[Safe Stringify] Error:', err.message);
        console.error('[Safe Stringify] Stack:', err.stack);
        // If still fails, return minimal error information
        try {
            return JSON.stringify({
                error: 'Failed to serialize response',
                message: err.message,
                type: 'SerializationError'
            });
        } catch (fallbackErr) {
            // Last resort: return a simple string
            return '{"error":"Critical serialization failure"}';
        }
    }
}

// Safe JSON response helper
function safeJsonResponse(res, obj, statusCode = 200) {
    try {
        const serialized = safeStringify(obj);
        res.status(statusCode).type('json').send(serialized);
    } catch (err) {
        console.error('[Safe Json Response] Critical error:', err);
        if (!res.headersSent) {
            res.status(500).json({ 
                error: 'Failed to send response',
                message: err.message 
            });
        }
    }
}

// Health check endpoint
app.get("/health", (req, res) => {
    res.json({ status: "ok" });
});

app.post("/stop", (req, res) => {
    if (bot) {
        if (bot.viewer) {
            bot.viewer.close();
        }
        bot.end();
    }
    res.json({
        message: "Bot stopped",
    });
});

// Counts SERVER ticks over a fixed window via world age (advances only when
// the server ticks; client-side physicsTick fires regardless of server pause).
// Read-only — no chat, no state change. Used by bridge to probe whether MC is
// actually ticking (i.e. unpaused) without sending a /pause toggle that could
// re-pause an already-unpaused server.
app.post("/observe_ticks", async (req, res) => {
    if (!bot) {
        res.status(400).json({ error: "Bot not spawned" });
        return;
    }
    const timeoutMs = (req.body && Number.isFinite(req.body.timeout_ms))
        ? Math.max(500, Math.min(15000, req.body.timeout_ms))
        : 3000;
    const ageOf = () => (bot.time ? bot.time.age : 0);
    const a0 = ageOf();
    await new Promise((resolve) => setTimeout(resolve, timeoutMs));
    res.json({ ticks: ageOf() - a0 });
});

app.post("/pause", async (req, res) => {
    if (!bot) {
        res.status(400).json({ error: "Bot not spawned" });
        return;
    }
    // /pause is a toggle. Confirm the target state via world age, which advances
    // only when the server ticks: unpause returns once age starts advancing,
    // pause once it stops, instead of waiting the full timeout.
    const expectRunning = req.body && req.body.expect_running === true;
    const timeoutMs = (req.body && Number.isFinite(req.body.timeout_ms))
        ? Math.max(500, Math.min(15000, req.body.timeout_ms))
        : 5000;
    const ageOf = () => (bot.time ? bot.time.age : 0);
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    const _pauseT0 = Date.now();
    bot.chat("/pause");
    let confirmed = false;
    if (expectRunning) {
        const base = ageOf();
        while (Date.now() - _pauseT0 < timeoutMs) {
            await sleep(25);
            if (ageOf() > base) { confirmed = true; break; }
        }
    } else {
        const STILL_MS = 250; // age unchanged this long => server frozen
        let prev = ageOf(); let lastChange = Date.now();
        while (Date.now() - _pauseT0 < timeoutMs) {
            await sleep(25);
            const a = ageOf();
            if (a !== prev) { prev = a; lastChange = Date.now(); }
            else if (Date.now() - lastChange >= STILL_MS) { confirmed = true; break; }
        }
    }
    const elapsed = Date.now() - _pauseT0;
    if (expectRunning && !confirmed) {
        return res.status(503).json({ error: "server did not resume ticking", ms: elapsed });
    }
    res.json({ message: "Success", confirmed: confirmed, ms: elapsed });
});

// ============================================================
//  Client-side mcpr recorder (records EVERY packet the server
//  sent to the bot, including per-player SetContainerSlot /
//  SetContainerContent — which ServerReplay drops because it
//  records broadcasts only). Output is a stock mcpr v14 file
//  loadable in ReplayMod or a compatible replay renderer.
//
//  Note: McprRecorder is created and `startBuffering()` is called
//  inside the /start handler — synchronously, right after
//  `mineflayer.createBot()` — so the LOGIN handshake (incl. the
//  LoginSuccess packet that replaystudio's reader keys off of to
//  transition LOGIN → PLAY) lands at the head of the recording.
//  /start_mcpr below just supplies the output path and calls
//  commit() to flush the buffered packets to disk.
// ============================================================

app.post("/start_mcpr", (req, res) => {
    if (!bot) {
        res.status(400).json({ error: "Bot not started" });
        return;
    }
    const out = req.body.output;
    if (!out) {
        res.status(400).json({ error: "Missing required parameter: output (path to .mcpr)" });
        return;
    }
    if (!mcprRecorder) {
        // Bot started without buffering (shouldn't happen post-fix, but
        // keep a graceful path). Do a fresh single-phase start — accepts
        // that the LOGIN handshake is already gone.
        try {
            mcprRecorder = new McprRecorder(bot, out);
            mcprRecorder.start();
            res.json({ status: "mcpr_recording_started_legacy", output: out });
        } catch (e) {
            console.error("[start_mcpr] legacy start failed:", e);
            mcprRecorder = null;
            res.status(500).json({ error: e.message });
        }
        return;
    }
    if (mcprRecorder.outputPath) {
        res.status(400).json({ error: "mcpr recording already committed",
                                output: mcprRecorder.outputPath });
        return;
    }
    try {
        mcprRecorder.commit(out);
        res.json({ status: "mcpr_recording_committed", output: out });
    } catch (e) {
        console.error("[start_mcpr] commit failed:", e);
        res.status(500).json({ error: e.message });
    }
});

app.post("/stop_mcpr", async (req, res) => {
    if (!mcprRecorder) {
        res.status(400).json({ error: "No mcpr recording in progress" });
        return;
    }
    const out = mcprRecorder.outputPath;
    try {
        await mcprRecorder.stop();
        mcprRecorder = null;
        res.json({ status: "mcpr_recording_stopped", output: out });
    } catch (e) {
        console.error("[stop_mcpr] failed:", e);
        mcprRecorder = null;
        res.status(500).json({ error: e.message });
    }
});

// On graceful shutdown (SIGTERM from the supervising process, or SIGINT
// from Ctrl-C), finalize any in-flight mcpr so it lands on disk as a
// proper .mcpr file instead of an orphaned tmpdir.
async function finalizeMcprOnExit(signal) {
    if (mcprRecorder) {
        try {
            console.log(`[${signal}] finalizing in-flight mcpr...`);
            await mcprRecorder.stop();
            mcprRecorder = null;
        } catch (e) {
            console.error(`[${signal}] mcpr stop failed:`, e);
        }
    }
    process.exit(0);
}
process.on('SIGTERM', () => finalizeMcprOnExit('SIGTERM'));
process.on('SIGINT',  () => finalizeMcprOnExit('SIGINT'));

// Server listening to PORT 3000

const DEFAULT_PORT = 3000;
const PORT = process.argv[2] || DEFAULT_PORT;
app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});
