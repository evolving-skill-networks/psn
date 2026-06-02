mineBlock(bot, name, count = 1)
  Mine blocks by name and collect dropped items into inventory.
  name: block name string (must match mcData.blocksByName).
  count: how many blocks to mine.
  Returns number of items actually gained. Returns 0 if no blocks found within 32 blocks.
  Handles pathfinding to blocks, tool selection, mining, and item collection internally.
