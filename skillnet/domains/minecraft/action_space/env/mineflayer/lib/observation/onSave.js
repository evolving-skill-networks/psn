const Observation = require("./base.js").Observation;

class onSave extends Observation {
    constructor(bot) {
        super(bot);
        this.name = "onSave";
        this.obs = [];
        this.maxLen = 50;
        bot.on("save", (eventName) => {
            // Save entity status to local variable
            if (typeof eventName === "string" && eventName.length > 0) {
                this.obs.push(eventName);
                if (this.obs.length > this.maxLen) this.obs.shift();
            }
            this.bot.event(this.name);
        });
    }

    observe() {
        const result = this.obs.join("\n");
        this.obs = [];
        return result;
    }
}

module.exports = onSave;
