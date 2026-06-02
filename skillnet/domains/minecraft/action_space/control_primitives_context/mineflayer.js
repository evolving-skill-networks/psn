Mineflayer API reference:

PATHFINDER:
await bot.pathfinder.goto(goal)  — navigate to goal. May change equipped item.

GOAL TYPES:
new GoalNear(x, y, z, range)         — move within range of position
new GoalXZ(x, z)                     — horizontal goal (ignores Y)
new GoalGetToBlock(x, y, z)          — get adjacent to block (for interaction)
new GoalFollow(entity, range)        — follow entity within range
new GoalPlaceBlock(position, bot.world, {})  — position to place a block
new GoalLookAtBlock(position, bot.world, {}) — position to see a block face

BLOCK SEARCH (two different functions — do not confuse):
bot.findBlock({matching, maxDistance})          — returns SINGLE Block or null (nearest match only)
bot.findBlocks({matching, maxDistance, count})  — returns Block[] array (up to count matches)
  WARNING: bot.findBlock (singular) returns ONE block or null, NOT an array.
  Do NOT call .length on its result. Use bot.findBlocks (plural) when you need multiple blocks.

UTILITY:
bot.blockAt(position)           — returns Block at Vec3 position, or null if chunk not loaded
bot.nearestEntity(filter)       — find nearest entity matching filter function

ASYNC FUNCTIONS:
await bot.equip(item, destination)  — equip item to slot ("hand", "head", "torso", "legs", "feet", "off-hand")
await bot.consume()             — consume held item (food, potion)
await bot.fish()                — fish (need water + fishing rod equipped)
await bot.sleep(bedBlock)       — sleep until sunrise
await bot.activateBlock(block)  — right-click block (buttons, doors)
await bot.lookAt(position)      — look at Vec3 position (need to be nearby)
await bot.activateItem()        — right-click to use held item (buckets)
await bot.useOn(entity)         — right-click entity (shearing, harness)
