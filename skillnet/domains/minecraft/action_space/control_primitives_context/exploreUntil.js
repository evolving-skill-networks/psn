exploreUntil(bot, direction, maxTime = 60, callback = () => false)
  Walk in a direction until callback returns a truthy value, or maxTime seconds elapse.
  direction: Vec3 with components -1, 0, or 1 only (e.g., new Vec3(1, 0, 0) for east).
  maxTime: seconds (max 1200).
  callback: called every 2 seconds, return truthy to stop.
  Returns callback's return value if found, null if timeout.
  Checks callback before moving. Handles pathfinding and cleanup internally.
