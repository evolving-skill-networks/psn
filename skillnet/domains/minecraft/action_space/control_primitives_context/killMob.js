killMob(bot, mobName, timeout = 300)
  Find, kill a mob, and collect dropped items.
  mobName: entity name string (must match entity.name).
  timeout: max seconds to wait for mob death.
  Searches within 48 blocks. If no mob found, returns without throwing.
  Auto-detects weapon type (melee vs ranged) and uses appropriate attack method.
  Handles combat, item collection, and cleanup internally.
