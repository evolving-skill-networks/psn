// Phase 13.B-5: Spatial neighborhood observer.
// Provides raw block.name for every integer position within a small radius
// around the bot's foot, so the LLM can derive placement candidates from
// truthful per-cell data (no candidate-set hint).
const { Observation } = require("./base");

class Spatial extends Observation {
    constructor(bot) {
        super(bot);
        this.name = "spatial";
        // Phase 13.B-6 D-test: radius=2 (5x5x5=125 cells, ~3.4KB prompt)
        // outperformed radius=3 (~10KB) for qwen3 placement reliability:
        //   shaft: 9/10 placed (vs 7/10 at r=3)
        //   box:   6/10 canonical (vs 1/10 at r=3)
        // Smaller radius reduces prompt overhead AND improves canonical
        // preference because there's less outside-trick "air" visible.
        this.radius = 2;
    }

    observe() {
        const r = this.radius;
        const out = [];
        if (!this.bot || !this.bot.entity) return out;
        const foot = this.bot.entity.position.floored();
        for (let dx = -r; dx <= r; dx++) {
            for (let dy = -r; dy <= r; dy++) {
                for (let dz = -r; dz <= r; dz++) {
                    const block = this.bot.blockAt(foot.offset(dx, dy, dz));
                    if (!block) continue;
                    out.push({
                        x: foot.x + dx,
                        y: foot.y + dy,
                        z: foot.z + dz,
                        name: block.name,
                    });
                }
            }
        }
        return out;
    }
}

module.exports = Spatial;
