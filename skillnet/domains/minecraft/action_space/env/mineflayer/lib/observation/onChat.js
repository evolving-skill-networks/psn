const Observation = require("./base.js").Observation;

class onChat extends Observation {
    constructor(bot) {
        super(bot);
        this.name = "onChat";
        this.obs = [];
        this.maxLen = 100;
        bot.on("chatEvent", (username, message) => {
            // Save entity status to local variable
            if (message.startsWith("/")) {
                return;
            }

            if (typeof message === "string" && message.length > 0) {
                this.obs.push(message);
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

module.exports = onChat;
