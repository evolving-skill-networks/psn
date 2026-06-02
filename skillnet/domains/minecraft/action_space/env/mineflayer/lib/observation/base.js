class Observation {
    constructor(bot) {
        if (new.target === Observation) {
            throw new TypeError(
                "Cannot instantiate abstract class Observation"
            );
        }

        this.bot = bot;
        this.name = "Observation";
    }

    observe() {
        throw new TypeError("Method 'observe()' must be implemented.");
    }

    reset() {}
}

function inject(bot, obs_list) {
    bot.obsList = [];
    bot.cumulativeObs = [];
    bot.eventMemory = {};
    obs_list.forEach((obs) => {
        bot.obsList.push(new obs(bot));
    });
    bot.event = function (event_name) {
        let result = {};
        bot.obsList.forEach((obs) => {
            if (obs.name.startsWith("on") && obs.name !== event_name) {
                return;
            }
            try {
                result[obs.name] = obs.observe();
            } catch (err) {
                // If an observation fails, log the error but do not break the whole flow
                console.error(`[Observation Error] Failed to observe ${obs.name}: ${err.message}`);
                if (err.stack) {
                    console.error(`[Observation Error] Stack: ${err.stack}`);
                }
                // Set a default value for the failed observation so the result object still has the key
                // This way the Python side does not error out due to a missing key
                if (obs.name === "inventory") {
                    result[obs.name] = {}; // inventory defaults to empty object
                } else if (obs.name === "status") {
                    result[obs.name] = {}; // status defaults to empty object
                } else if (obs.name === "voxels") {
                    result[obs.name] = []; // voxels defaults to empty array
                } else {
                    result[obs.name] = null; // others default to null
                }
            }
        });
        bot.cumulativeObs.push([event_name, result]);
    };
    bot.observe = function () {
        bot.event("observe");
        const result = bot.cumulativeObs;
        bot.cumulativeObs = [];
        return JSON.stringify(result);
    };
}

module.exports = { Observation, inject };
