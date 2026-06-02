placeItem(bot, name, position)
  Place an inventory item as a block at the given position.
  name: item name string.
  position: Vec3 — a valid placement position requires ALL of:
    (1) bot.blockAt(position) is air or replaceable (water, tall_grass)
    (2) At least one face-adjacent block (±x, ±y, ±z) is NOT air (serves as placement reference)
    (3) Position is NOT occupied by the bot entity (causes timeout)
  Fails with "no valid reference block" if condition (2) is not met.
  Handles navigation to placement position and block placement internally.
  Throws on failure (do not suppress with try-catch).
