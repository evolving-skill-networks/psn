# Fabric Mods for PSN

PSN runs on a Minecraft 1.19.4 client with Fabric Loader installed. Beyond Fabric Loader itself, PSN needs **two mods** at the client/server to behave correctly. Two more are optional dev-quality-of-life additions.

If you set up the headless Fabric server via `./skillnet.sh --mc-mode=headless`, you do not need to install these manually — the headless server is unmodded and PSN's pause command is a no-op there. The mods below matter only for the **Modrinth / desktop client** path.

---

## Required

### 1. Fabric API (1.19.4)

Almost every Fabric mod (including the next one) depends on this. Without it Fabric Loader refuses to start the modded ones.

- Download: https://modrinth.com/mod/fabric-api/versions?g=1.19.4&l=fabric
- Drop the JAR into your instance's `mods/` directory.

### 2. Multiplayer Server Pause (1.19.4)

PSN sends `/pause` chat commands to halt server ticks while the LLM thinks (each call takes 5-30s). Without this mod the world keeps ticking — mobs despawn, day/night cycles advance, hunger drains, and the bot's planned actions silently miss their targets. **Strongly recommended** even though PSN still technically runs without it.

- Download: https://modrinth.com/mod/multiplayer-server-pause/versions?g=1.19.4&l=fabric
- This mod has a **runtime dependency** on iChunUtil:
  - https://modrinth.com/mod/ichunutil/versions?g=1.19.4&l=fabric

Drop both JARs into `mods/`. (Note: older 1.19 builds of Multiplayer Server Pause depended on `CompleteConfig`; the 1.19.4 build, version 1.3.x, depends on iChunUtil instead.)

---

## Optional

### Mod Menu

A pause-screen UI that lists installed mods and surfaces their config screens. Convenience-only; PSN does not care whether it is installed.

- https://modrinth.com/mod/modmenu/versions?g=1.19.4&l=fabric

### Server Replay (Fabric)

Records every packet the server sent to a player into a `.mcpr` file that you can play back in any Minecraft client. Useful for diagnosing what the bot saw during a particular iteration. PSN's `PSN_MCPR_OUTPUT` env var (see `.env.example`) enables a parallel **client-side** recording without this mod — install Server Replay only if you want full server-side packet captures.

- https://modrinth.com/mod/server-replay/versions?g=1.19.4&l=fabric
- Runtime dependency: [Fabric Language Kotlin](https://modrinth.com/mod/fabric-language-kotlin/versions?g=1.19.4&l=fabric).

---

## Installation (Modrinth instance)

If you followed [Path A in the main README](../README.md#path-a--minecraft-client-via-modrinth-recommended), open your Modrinth instance and add mods through the GUI:

1. In Modrinth, click your PSN instance → **Mods** tab → **Add content**.
2. Search for each mod name above, pick the **1.19.4 / Fabric** build, click install. Modrinth handles dependencies for you (so installing Multiplayer Server Pause should also pull in iChunUtil automatically — verify the dep is listed and present).
3. Launch the instance once to confirm all mods load (you should see them on the title screen if Mod Menu is also installed).

If your launcher is not Modrinth, locate the `mods/` directory of the instance and drop the JARs there manually.
