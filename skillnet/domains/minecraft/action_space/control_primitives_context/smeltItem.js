smeltItem(bot, itemName, fuelName, count = 1)
  Smelt items in a furnace.
  itemName: item to smelt (must exist in mcData.itemsByName).
  fuelName: fuel item (must be a valid fuel).
  count: how many items to smelt.
  Requires a placed furnace block within 32 blocks.
  A furnace ITEM in inventory is NOT enough — it must be placed as a block first using placeItem.
  Searches for furnace block and navigates to it internally.
  Both itemName and fuelName must be in inventory.
  Throws on failure (do not suppress with try-catch).
