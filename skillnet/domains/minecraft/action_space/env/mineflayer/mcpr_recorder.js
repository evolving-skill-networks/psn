/**
 * Client-side mcpr recorder for the PSN mineflayer bot.
 *
 * Why this exists: ServerReplay (the existing server-side recorder) only
 * captures broadcast packets, so per-player packets like SetContainerSlot
 * (0x14) and SetContainerContent (0x12 = window_items) — which carry the
 * bot's actual inventory state over time — are dropped. This recorder runs
 * INSIDE the bot's mineflayer client, so it sees every packet the server
 * sent specifically to the bot, including all inventory updates.
 *
 * Output is a stock mcpr v14 file (zip with metaData.json +
 * recording.tmcpr + recording.tmcpr.crc32) that ReplayMod can load and
 * scrub like any other recording.
 *
 * Two-phase usage (preferred — captures the LOGIN handshake so
 * replaystudio's ReplayInputStream can transition LOGIN → PLAY naturally):
 *     // Right after `bot = mineflayer.createBot(...)`:
 *     const rec = new McprRecorder(bot);
 *     rec.startBuffering();
 *     // ... time passes; LOGIN handshake completes; bot spawns ...
 *     rec.commit('/path/to/output.mcpr');  // flush buffer + continue live
 *     // ... later ...
 *     await rec.stop();
 *
 * Self-player injection
 * ---------------------
 * In a vanilla Minecraft connection, the server does NOT send the local
 * player's own player profile, spawn packet, or movement updates back to
 * that player (you don't see yourself as an entity). So a stock client-
 * side mcpr lacks any trace of the bot's player entity in the world,
 * which means ReplayMod can never resolve `spectate_username: "bot"` and
 * the renderer skips every video.
 *
 * To make the bot a spectatable, moving entity in the replay, this
 * recorder synthesizes the missing packets and weaves them into the
 * stream:
 *   • PlayerInfoUpdate (0x3a, action=add_player) — adds the bot's
 *     UUID+name to the client's tab list once, right after JoinGame.
 *   • SpawnPlayer (0x03 named_entity_spawn) — spawns the bot's player
 *     entity at its initial position once, right after JoinGame.
 *   • EntityTeleport (0x68) — fires on every mineflayer 'move' event,
 *     mirroring bot.entity.position+yaw+pitch onto the bot's player
 *     entity so it tracks the bot's actual movement during replay.
 *
 * Without this injection, the renderer's spectate-by-username lookup
 * always returns null → render aborts → no MP4.
 */

const fs = require('fs');
const path = require('path');
// archiver v8 changed its export: it's no longer a callable
// `archiver(format, opts)` factory but a namespace object
// `{ Archiver, ZipArchive, ... }`. Use the ZipArchive class directly.
const { ZipArchive } = require('archiver');
const crc32 = require('buffer-crc32').unsigned;

const PROTOCOL = 762;       // 1.19.4
const MC_VERSION = '1.19.4';
const FILE_FORMAT_VERSION = 14;

// Packet IDs we care about for self-player injection (protocol 762, PLAY
// clientbound). Sourced from minecraft-data 1.19.4/protocol.json.
const PID_NAMED_ENTITY_SPAWN  = 0x03;   // = SpawnPlayer
const PID_JOIN_GAME           = 0x28;
const PID_PLAYER_INFO         = 0x3a;   // = PlayerInfoUpdate
const PID_ENTITY_HEAD_ROTATION = 0x42;  // = EntityHeadLook (head yaw only)
const PID_ENTITY_TELEPORT     = 0x68;

// MC's 1-byte angle: 256 = full turn. mineflayer reports yaw/pitch in
// radians; convert and wrap.
function radiansToByteAngle(radians) {
    const n = Math.round(radians * 128 / Math.PI) & 0xFF;
    return n;
}

// Minimum varint encoder — packet ids and lengths are < 0x80 here, but
// keep the loop for safety in case data_count > 127 someday.
function writeVarInt(value) {
    const out = [];
    let v = value >>> 0;
    while (true) {
        if ((v & ~0x7F) === 0) {
            out.push(v);
            return Buffer.from(out);
        }
        out.push((v & 0x7F) | 0x80);
        v >>>= 7;
    }
}

function writeString(s) {
    const b = Buffer.from(s, 'utf8');
    return Buffer.concat([writeVarInt(b.length), b]);
}

// Convert a hyphenated UUID string ("67128b5b-2e6b-3ad1-baa0-1b937b03e5c5")
// to the raw 16-byte big-endian buffer Minecraft expects on the wire.
function uuidStringToBytes(uuidStr) {
    const hex = String(uuidStr).replace(/-/g, '');
    if (hex.length !== 32) {
        throw new Error(`unexpected UUID format: ${uuidStr}`);
    }
    return Buffer.from(hex, 'hex');
}

// PLAY 0x3a player_info, action bitflags = add_player only, exactly one
// data entry: <UUID 16><string name><varint properties_count=0>
function buildPlayerInfoAdd(uuidBytes, name) {
    return Buffer.concat([
        writeVarInt(PID_PLAYER_INFO),
        Buffer.from([0x01]),                   // action bitflags = add_player
        writeVarInt(1),                        // data array length
        uuidBytes,                             // UUID
        writeString(name),                     // game_profile.name
        writeVarInt(0),                        // game_profile.properties (empty)
    ]);
}

// PLAY 0x03 named_entity_spawn (SpawnPlayer):
//   <varint pid><varint entityId><UUID><f64 x><f64 y><f64 z><i8 yaw><i8 pitch>
function buildSpawnPlayer(entityId, uuidBytes, x, y, z, yawRad, pitchRad) {
    const pos = Buffer.alloc(24);
    pos.writeDoubleBE(x, 0);
    pos.writeDoubleBE(y, 8);
    pos.writeDoubleBE(z, 16);
    return Buffer.concat([
        writeVarInt(PID_NAMED_ENTITY_SPAWN),
        writeVarInt(entityId),
        uuidBytes,
        pos,
        Buffer.from([radiansToByteAngle(yawRad), radiansToByteAngle(pitchRad)]),
    ]);
}

// PLAY 0x68 entity_teleport:
//   <varint pid><varint entityId><f64 x><f64 y><f64 z><i8 yaw><i8 pitch><bool onGround>
function buildEntityTeleport(entityId, x, y, z, yawRad, pitchRad, onGround) {
    const pos = Buffer.alloc(24);
    pos.writeDoubleBE(x, 0);
    pos.writeDoubleBE(y, 8);
    pos.writeDoubleBE(z, 16);
    return Buffer.concat([
        writeVarInt(PID_ENTITY_TELEPORT),
        writeVarInt(entityId),
        pos,
        Buffer.from([radiansToByteAngle(yawRad),
                     radiansToByteAngle(pitchRad),
                     onGround ? 1 : 0]),
    ]);
}

// PLAY 0x42 entity_head_rotation:
//   <varint pid><varint entityId><i8 headYaw>
// EntityTeleport's yaw sets the entity's BODY yaw; the camera that
// ReplayMod's first-person spectator follows uses the entity's HEAD
// yaw, which only updates via this packet. Without it, the rendered
// video shows the bot walking but the camera never pans.
function buildEntityHeadRotation(entityId, headYawRad) {
    return Buffer.concat([
        writeVarInt(PID_ENTITY_HEAD_ROTATION),
        writeVarInt(entityId),
        Buffer.from([radiansToByteAngle(headYawRad)]),
    ]);
}

class McprRecorder {
    constructor(bot, outputPath, opts = {}) {
        this.bot = bot;
        this.outputPath = outputPath;       // may be null at construction
        this.opts = opts;
        this.tmpDir = null;
        this.tmcprPath = null;
        this.startWallclockMs = null;
        this.tmcprStream = null;
        this._handler = null;
        this._earlyHandler = null;
        // In-flight in-memory buffer used between startBuffering() and
        // commit() — captures LOGIN-handshake packets that arrive before
        // we know the output path. Each entry: {elapsed, rawBuffer, meta}.
        this._earlyBuffer = null;
        this._stopped = false;
        // Self-player injection state (see file header). Populated lazily
        // as JoinGame and 'spawn' arrive.
        this._botEntityId = null;
        this._spawnPlayerInjected = false;
        this._moveListener = null;
        // Throttle move-event emissions so we don't write 100+ teleports
        // per second when the bot's physics ticks rapidly. MC's tick is
        // 50ms; one mirror per tick is plenty.
        this._lastMoveWriteMs = 0;
    }

    // --- Two-phase API -----------------------------------------------------

    /**
     * Attach the 'raw' listener to capture every parsed inbound packet
     * (handshake + login + play + configuration). Packets are accumulated
     * in memory until commit() supplies an output path. Call this
     * IMMEDIATELY after `mineflayer.createBot()` so the LOGIN handshake
     * lands at the head of the recording — replaystudio's ReplayInputStream
     * needs the LoginSuccess packet to transition its rawRegistry from
     * LOGIN state into PLAY state, otherwise PLAY packets get parsed in
     * LOGIN context and ReplayMod throws "Packet 2/0 was larger than I
     * expected, found N bytes extra".
     */
    startBuffering() {
        if (this.startWallclockMs !== null) return;
        this.startWallclockMs = Date.now();
        this._earlyBuffer = [];
        this._earlyHandler = (rawBuffer, meta) => {
            if (McprRecorder._isReplayHostilePacket(meta)) return;
            // Buffer.from() copies — NMP may reuse the underlying buffer
            // across packets, so we'd lose data without a copy.
            this._earlyBuffer.push({
                elapsed: Date.now() - this.startWallclockMs,
                rawBuffer: Buffer.from(rawBuffer),
                meta,
            });
            this._maybeCaptureEntityIdFromJoinGame(rawBuffer, meta);
        };
        this.bot._client.on('raw', this._earlyHandler);
        console.log('[mcpr_recorder] buffering early packets (no output path yet)…');
    }

    /**
     * Connection-state-changing packets that ReplayMod's replay pipeline
     * does NOT want to see in an mcpr file:
     *
     *   • LOGIN.encryption_begin (0x01) — for the live wire only; the
     *     replay client never decrypts so this just confuses the pipeline.
     *
     *   • LOGIN.compress (SetCompression, 0x03) — activates the
     *     decompressor. We capture packets POST-decompression from NMP,
     *     so once the replay's decompressor activates it tries to inflate
     *     our already-decompressed bytes and throws
     *     `DataFormatException: incorrect header check`. ServerReplay also
     *     omits this packet for the same reason.
     */
    static _isReplayHostilePacket(meta) {
        if (!meta) return false;
        if (meta.state !== 'login') return false;
        return meta.name === 'compress' || meta.name === 'encryption_begin';
    }

    _maybeCaptureEntityIdFromJoinGame(rawBuffer, meta) {
        if (this._botEntityId !== null) return;
        if (!meta || meta.state !== 'play' || meta.name !== 'login') return;
        // NMP's PLAY packet 'login' = JoinGame (0x28 in protocol 762).
        // Wire format after pid: i32 entityId. The pid is 1 byte (0x28).
        if (rawBuffer.length < 5 || rawBuffer[0] !== PID_JOIN_GAME) return;
        try {
            this._botEntityId = rawBuffer.readInt32BE(1);
            console.log(`[mcpr_recorder] captured bot entityId=${this._botEntityId} from JoinGame`);
        } catch (e) {
            console.error('[mcpr_recorder] JoinGame entityId parse failed:', e.message);
        }
    }

    /**
     * Commit the buffered early packets to disk under outputPath, then
     * continue writing live packets straight to disk. Detaches the
     * buffering handler and installs the live-write handler.
     */
    commit(outputPath) {
        if (this._earlyBuffer === null) {
            throw new Error('commit(): startBuffering() was not called first');
        }
        this.outputPath = outputPath;
        this._openOutputFile();

        // Flush every packet captured during the buffering window.
        let flushed = 0;
        for (const e of this._earlyBuffer) {
            this._writePacket(e.elapsed, e.rawBuffer);
            flushed += 1;
        }
        this._earlyBuffer = null;

        // Swap to live-write handler.
        this.bot._client.removeListener('raw', this._earlyHandler);
        this._earlyHandler = null;
        this._installLiveHandler();
        // Wire up self-player injection now that we have a stream open.
        this._installSelfPlayerInjection();
        console.log(`[mcpr_recorder] committed → ${this.outputPath} ` +
                `(flushed ${flushed} buffered packets)`);
    }

    // --- Legacy single-phase API ------------------------------------------

    /**
     * Open the output file and attach the live-write 'raw' handler in one
     * step. Use this only when you don't need ReplayMod-compatibility
     * (e.g., you only intend to parse the recording with our Python
     * parsers); it WILL miss the LOGIN handshake and produces files
     * ReplayMod cannot replay.
     */
    start() {
        if (this.startWallclockMs !== null) return;
        if (!this.outputPath) {
            throw new Error('start(): outputPath is required for single-phase mode');
        }
        this._openOutputFile();
        this.startWallclockMs = Date.now();
        this._installLiveHandler();
        this._installSelfPlayerInjection();
        console.log(`[mcpr_recorder] started → ${this.outputPath}`);
    }

    // --- Internals --------------------------------------------------------

    _openOutputFile() {
        this.tmpDir = this.outputPath + '.tmpdir';
        this.tmcprPath = path.join(this.tmpDir, 'recording.tmcpr');
        try { fs.rmSync(this.tmpDir, { recursive: true, force: true }); } catch (_) {}
        fs.mkdirSync(this.tmpDir, { recursive: true });
        this.tmcprStream = fs.createWriteStream(this.tmcprPath);
    }

    _installLiveHandler() {
        // The 'raw' event fires for every parsed packet with (rawBuffer,
        // metadata) where rawBuffer is the on-the-wire packet bytes
        // (packet ID varint + payload, no length prefix, post-decompress
        // and post-decrypt). That's exactly mcpr's expected format per
        // packet entry — and we record across ALL connection states (no
        // state filter) so the LOGIN handshake survives in the file.
        // The only exclusions are connection-state-changing packets that
        // would mis-configure ReplayMod's playback pipeline (see
        // _isReplayHostilePacket).
        this._handler = (rawBuffer, meta) => {
            if (McprRecorder._isReplayHostilePacket(meta)) return;
            const elapsed = Date.now() - this.startWallclockMs;
            this._writePacket(elapsed, rawBuffer);
            this._maybeCaptureEntityIdFromJoinGame(rawBuffer, meta);
        };
        this.bot._client.on('raw', this._handler);
    }

    /**
     * Wire up the synthetic-packet injection that makes the bot appear as
     * a spectatable, moving entity in the replay. Two pieces:
     *
     *   1. On bot 'spawn', if we have an entityId from JoinGame, write
     *      one PlayerInfoUpdate (add_player) + one SpawnPlayer right then.
     *      'spawn' is the right hook because by then bot.entity.position
     *      is the actual world spawn position, not (0,0,0).
     *
     *   2. On bot 'move', mirror bot.entity.position+yaw+pitch into an
     *      EntityTeleport for the bot's player entity. Throttled to one
     *      write per tick (50ms) so we don't spam thousands of teleports
     *      per second when physicsTick fires rapidly.
     */
    _installSelfPlayerInjection() {
        const onSpawn = () => {
            if (this._spawnPlayerInjected) return;
            if (this._botEntityId === null) {
                console.warn('[mcpr_recorder] spawn fired but JoinGame entityId not captured yet — skipping SpawnPlayer injection');
                return;
            }
            const uuidStr = this.bot._client && this.bot._client.uuid;
            if (!uuidStr) {
                console.warn('[mcpr_recorder] no bot._client.uuid at spawn — skipping SpawnPlayer injection');
                return;
            }
            const username = (this.bot._client && this.bot._client.username) || 'bot';
            const ent = this.bot.entity;
            if (!ent) {
                console.warn('[mcpr_recorder] no bot.entity at spawn — skipping SpawnPlayer injection');
                return;
            }
            try {
                const uuidBytes = uuidStringToBytes(uuidStr);
                const elapsed = Date.now() - this.startWallclockMs;
                const piPkt = buildPlayerInfoAdd(uuidBytes, username);
                const spPkt = buildSpawnPlayer(this._botEntityId, uuidBytes,
                        ent.position.x, ent.position.y, ent.position.z,
                        ent.yaw || 0, ent.pitch || 0);
                this._writePacket(elapsed, piPkt);
                this._writePacket(elapsed, spPkt);
                this._spawnPlayerInjected = true;
                console.log(`[mcpr_recorder] injected synthetic PlayerInfo+SpawnPlayer for bot id=${this._botEntityId}`);
            } catch (e) {
                console.error('[mcpr_recorder] self-player injection failed:', e);
            }
        };
        // bot.spawn fires once per session. If it already fired (we wired
        // up after spawn), call onSpawn directly.
        if (this.bot.entity) {
            onSpawn();
        } else {
            this.bot.once('spawn', onSpawn);
        }

        // Mirror movement into entity_teleport packets. Throttle to one
        // per tick (50ms) so we cap at ~20Hz regardless of physicsTick
        // burstiness.
        this._moveListener = () => {
            if (!this._spawnPlayerInjected || this._botEntityId === null) return;
            const ent = this.bot.entity;
            if (!ent) return;
            const now = Date.now();
            if (now - this._lastMoveWriteMs < 45) return;  // ~22Hz cap
            this._lastMoveWriteMs = now;
            try {
                const elapsed = now - this.startWallclockMs;
                // EntityTeleport: position + body yaw + pitch.
                const tp = buildEntityTeleport(this._botEntityId,
                        ent.position.x, ent.position.y, ent.position.z,
                        ent.yaw || 0, ent.pitch || 0,
                        ent.onGround !== undefined ? !!ent.onGround : true);
                this._writePacket(elapsed, tp);
                // EntityHeadRotation: head yaw. Without this, ReplayMod's
                // first-person spectator camera doesn't pan when the bot
                // turns. We use bot.entity.yaw for both because mineflayer
                // bots keep their head and body aligned.
                const hr = buildEntityHeadRotation(this._botEntityId,
                        ent.yaw || 0);
                this._writePacket(elapsed, hr);
            } catch (e) {
                console.error('[mcpr_recorder] entity_teleport write failed:', e.message);
            }
        };
        this.bot.on('move', this._moveListener);
    }

    _writePacket(elapsed, rawBuffer) {
        const header = Buffer.alloc(8);
        header.writeUInt32BE(elapsed, 0);
        header.writeUInt32BE(rawBuffer.length, 4);
        try {
            this.tmcprStream.write(header);
            this.tmcprStream.write(rawBuffer);
        } catch (e) {
            console.error('[mcpr_recorder] write failed:', e.message);
        }
    }

    async stop() {
        if (this._stopped || this.startWallclockMs === null) return;
        this._stopped = true;
        try {
            if (this._earlyHandler) {
                this.bot._client.removeListener('raw', this._earlyHandler);
                this._earlyHandler = null;
            }
            if (this._handler) {
                this.bot._client.removeListener('raw', this._handler);
                this._handler = null;
            }
            if (this._moveListener) {
                this.bot.removeListener('move', this._moveListener);
                this._moveListener = null;
            }
        } catch (_) {}

        // If we never committed (no output path supplied), drop the buffer.
        if (this._earlyBuffer !== null && !this.outputPath) {
            console.log(`[mcpr_recorder] stop() with no commit() — discarding ` +
                    `${this._earlyBuffer.length} buffered packets`);
            this._earlyBuffer = null;
            return;
        }
        // If we have output path but were still buffering, do an emergency
        // commit so the data isn't lost.
        if (this._earlyBuffer !== null && this.outputPath) {
            console.log('[mcpr_recorder] stop() while buffering — emergency commit');
            this._openOutputFile();
            for (const e of this._earlyBuffer) {
                this._writePacket(e.elapsed, e.rawBuffer);
            }
            this._earlyBuffer = null;
        }

        await new Promise((res) => this.tmcprStream.end(res));
        const tmcprBytes = fs.readFileSync(this.tmcprPath);
        const crcHex = crc32(tmcprBytes).toString(16).padStart(8, '0');

        // Use the captured bot entityId as selfId so consumers (incl. our
        // PsnReplayRendererMod) can resolve "the recording's local
        // player" without scanning packets.
        const selfId = this._botEntityId !== null ? this._botEntityId : -1;
        const meta = {
            singleplayer: false,
            serverName: this.opts.serverName || 'PSN-server',
            customServerName: this.opts.customServerName || 'PSN-server',
            duration: Date.now() - this.startWallclockMs,
            date: this.startWallclockMs,
            mcversion: MC_VERSION,
            fileFormat: 'MCPR',
            fileFormatVersion: FILE_FORMAT_VERSION,
            protocol: PROTOCOL,
            generator: 'PSN-mineflayer-recorder',
            selfId: selfId,
            players: this.bot._client && this.bot._client.uuid
                ? [this.bot._client.uuid]
                : [],
        };
        fs.writeFileSync(path.join(this.tmpDir, 'metaData.json'),
                JSON.stringify(meta));
        fs.writeFileSync(path.join(this.tmpDir, 'recording.tmcpr.crc32'), crcHex);
        // Also write an empty server_replay_packs.json (parity with
        // ServerReplay output — some readers check for it).
        fs.writeFileSync(path.join(this.tmpDir, 'server_replay_packs.json'), '[]');

        // Zip everything into the .mcpr file (archiver v8 ZipArchive API)
        await new Promise((res, rej) => {
            const out = fs.createWriteStream(this.outputPath);
            const archive = new ZipArchive({ zlib: { level: 5 } });
            out.on('close', res);
            archive.on('error', rej);
            archive.pipe(out);
            archive.file(path.join(this.tmpDir, 'metaData.json'),  { name: 'metaData.json' });
            archive.file(path.join(this.tmpDir, 'recording.tmcpr'), { name: 'recording.tmcpr' });
            archive.file(path.join(this.tmpDir, 'recording.tmcpr.crc32'), { name: 'recording.tmcpr.crc32' });
            archive.file(path.join(this.tmpDir, 'server_replay_packs.json'), { name: 'server_replay_packs.json' });
            archive.finalize();
        });

        try { fs.rmSync(this.tmpDir, { recursive: true, force: true }); } catch (_) {}
        const sz = fs.statSync(this.outputPath).size;
        console.log(`[mcpr_recorder] stopped — wrote ${this.outputPath} ` +
                `(${(sz / 1024 / 1024).toFixed(1)} MB, ${meta.duration} ms, selfId=${selfId})`);
    }
}

module.exports = McprRecorder;
