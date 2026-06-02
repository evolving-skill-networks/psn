getItemFromChest(bot, chestPosition, itemsToGet)
  Withdraw items from a chest.
  chestPosition: Vec3 position of the chest block.
  itemsToGet: object mapping item names to counts (e.g., {itemName: count}).
  Skips items not found in chest. Handles navigation and chest interaction internally.

depositItemIntoChest(bot, chestPosition, itemsToDeposit)
  Deposit items into a chest.
  chestPosition: Vec3 position of the chest block.
  itemsToDeposit: object mapping item names to counts.
  Skips items not in inventory. Handles navigation and chest interaction internally.

checkItemInsideChest(bot, chestPosition)
  Open a chest, list its contents, then close it.
  chestPosition: Vec3 position of the chest block.
