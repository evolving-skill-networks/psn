placeItem(bot, name, position)
  Place an inventory item as a block at the given position.
  name: item name string.
  position: Vec3 target position. Do not place at bot's own position (causes timeout).
  Handles position validation, reference block search in all 6 directions, pathfinding, and adaptive fallback internally.
  Throws on failure (do not suppress with try-catch).
