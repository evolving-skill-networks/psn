craftItem(bot, name, count = 1)
  Craft items. Handles recipe validation and execution internally.
  name: item name string (must exist in mcData.itemsByName).
  count: number of RECIPE EXECUTIONS (not output items).
  Each execution consumes inputs and produces a fixed number of outputs.
  Check the recipe output quantity before choosing count.
  3x3 recipes require a placed crafting_table block within 32 blocks.
  2x2 recipes work without a crafting table.
  Searches for a PLACED crafting table and navigates to it internally.
  craftItem does NOT place the crafting table for you. You must place it first.
  Throws on failure (do not suppress with try-catch).
