const Observation = require("./base.js").Observation;

class onError extends Observation {
    constructor(bot) {
        super(bot);
        this.name = "onError";
        this.obs = [];
        this.maxLen = 20;
        bot.on("error", (err) => {
            // Save entity status to local variable
            try {
                const msg = (err && err.message) ? err.message : String(err);
                const stack = (err && err.stack) ? err.stack : "";
                const combined = stack ? `${msg}\n${stack}` : msg;
                this.obs.push(combined);
                if (this.obs.length > this.maxLen) this.obs.shift();
            } catch (e) {
                this.obs.push("Unknown error");
                if (this.obs.length > this.maxLen) this.obs.shift();
            }
            this.bot.event(this.name);
        });
    }

    observe() {
        const result = this.obs.join("\n\n");
        this.obs = [];
        return result;
    }
}

module.exports = onError;
