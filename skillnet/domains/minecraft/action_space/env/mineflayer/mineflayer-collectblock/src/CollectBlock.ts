import { Bot } from 'mineflayer'
import { Block } from 'prismarine-block'
import { Movements, goals } from 'mineflayer-pathfinder'
import { TemporarySubscriber } from './TemporarySubscriber'
import { Entity } from 'prismarine-entity'
import { error } from './Util'
import { Vec3 } from 'vec3'
import { emptyInventoryIfFull, ItemFilter } from './Inventory'
import { findFromVein } from './BlockVeins'
import { Collectable, Targets } from './Targets'
import { Item } from 'prismarine-item'
import mcDataLoader from 'minecraft-data'
import { once } from 'events'
import { callbackify } from 'util'

export type Callback = (err?: Error) => void

async function collectAll (
  bot: Bot,
  options: CollectOptionsFull
): Promise<void> {
  let success_count = 0
  while (!options.targets.empty) {
    // PATCH (skillnet): cooperative cancellation point (see Targets.cancelled).
    if (options.targets.cancelled) break
    await emptyInventoryIfFull(
      bot,
      options.chestLocations,
      options.itemFilter
    )
    const closest = options.targets.getClosest()
    if (closest == null) break
    switch (closest.constructor.name) {
      case 'Block': {
        try {
          if (success_count >= options.count) {
            break
          }
          await bot.tool.equipForBlock(
            closest as Block,
            equipToolOptions
          )
          // PATCH (skillnet): a cancel can land while equipForBlock is awaited;
          // the goto below would then call setGoal, which DISCARDS the pending
          // pathfinder stop (stopPathing = false), resurrecting the task for a
          // whole extra target leg. Re-check before re-arming the pathfinder.
          if (options.targets.cancelled) break
          const goal = new goals.GoalLookAtBlock(
            closest.position,
            bot.world
          )
          await bot.pathfinder.goto(goal)
          await mineBlock(bot, closest as Block, options)
          success_count++
          // TODO: options.ignoreNoPath
        } catch (err) {
          // PATCH (skillnet): when the task was cancelled, a PathStopped /
          // GoalChanged rejection is the cancellation itself: propagate it out
          // of the loop instead of skip-one-continue, and do it BEFORE the
          // setGoal(null) below so a dying task never clobbers a goal the new
          // owner may already have set. Gated on cancelled so a mid-collect
          // path stop from elsewhere (e.g. the wedge-escape cap) keeps today's
          // per-target recovery.
          if (options.targets.cancelled &&
              ((err as any).name === 'PathStopped' || (err as any).name === 'GoalChanged')) {
            throw err
          }
          // console.log(err.stack)
          // bot.pathfinder.stop()
          // bot.waitForTicks(10)
          try {
            bot.pathfinder.setGoal(null)
          } catch (err) {}
          if (options.ignoreNoPath) {
            // @ts-expect-error
            if (err.name === 'Invalid block') {
              console.log(
                                `Block ${closest.name} at ${closest.position} is not valid! Skip it!`
              )
            } // @ts-expect-error
            else if (err.name === 'Unsafe block') {
              console.log(
                                `${closest.name} at ${closest.position} is not safe to break! Skip it!`
              )
              // @ts-expect-error
            } else if (err.name === 'Unbreakable') {
              console.log(
                                `${closest.name} at ${closest.position} is unbreakable! Skip it!`
              )
              // @ts-expect-error
            } else if (err.name === 'NoItem') {
              const properties =
                                bot.registry.blocksByName[closest.name as string]
              const leastTool = Object.keys(
                // @ts-expect-error
                properties.harvestTools
              )[0]
              // @ts-expect-error
              const item = bot.registry.items[leastTool]
              bot.chat(
                                `I need at least a ${item.name} to mine ${closest.name}!  Skip it!`
              )
              return
            } else if (
            // @ts-expect-error
              err.name === 'NoPath' ||
                            // @ts-expect-error
                            err.name === 'Timeout'
            ) {
              // P0: defensive check - bot.entity may be undefined on death/disconnect
              if (
                bot.entity &&
                                bot.entity.position.distanceTo(
                                  closest.position
                                ) < 0.5
              ) {
                await mineBlock(bot, closest as Block, options)
                break
              }
              console.log(
                                `No path to ${closest.name} at ${closest.position}! Skip it!`
              )
              // @ts-expect-error
            } else if (err.message === 'Digging aborted') {
              console.log('Digging aborted! Skip it!')
            } else {
              // @ts-expect-error
              bot.chat(`Error: ${err.message}`)
            }
            break
          }
          throw err
        }
        break
      }
      case 'Entity': {
        // Don't collect any entities that are marked as 'invalid'
        if (!(closest as Entity).isValid) break
        try {
          const tempEvents = new TemporarySubscriber(bot)
          const itemEntity = closest as Entity
          const itemStartPos = itemEntity.position.clone()
          const botStartPos = bot.entity.position.clone()

          // Diagnostic: Log pickup attempt start
          console.log(`[CollectBlock] Attempting to pick up: ${itemEntity.name || 'item'}`)
          console.log(`[CollectBlock]   Item pos: (${itemStartPos.x.toFixed(2)}, ${itemStartPos.y.toFixed(2)}, ${itemStartPos.z.toFixed(2)})`)
          console.log(`[CollectBlock]   Bot pos:  (${botStartPos.x.toFixed(2)}, ${botStartPos.y.toFixed(2)}, ${botStartPos.z.toFixed(2)})`)

          const waitForPickup = new Promise<void>(
            (resolve, reject) => {
              const timeout = setTimeout(() => {
                // After 10 seconds, reject the promise
                clearTimeout(timeout)
                tempEvents.cleanup()

                // Diagnostic: Log timeout with position info
                const botEndPos = bot.entity.position
                const itemEndPos = itemEntity.isValid ? itemEntity.position : null
                console.log('[CollectBlock] ❌ Pickup TIMEOUT after 10s')
                console.log(`[CollectBlock]   Bot final pos: (${botEndPos.x.toFixed(2)}, ${botEndPos.y.toFixed(2)}, ${botEndPos.z.toFixed(2)})`)
                if (itemEndPos != null) {
                  const dx = Math.abs(botEndPos.x - itemEndPos.x)
                  const dy = Math.abs(botEndPos.y - itemEndPos.y)
                  const dz = Math.abs(botEndPos.z - itemEndPos.z)
                  console.log(`[CollectBlock]   Item still at: (${itemEndPos.x.toFixed(2)}, ${itemEndPos.y.toFixed(2)}, ${itemEndPos.z.toFixed(2)})`)
                  console.log(`[CollectBlock]   Distance: dx=${dx.toFixed(2)}, dy=${dy.toFixed(2)}, dz=${dz.toFixed(2)}`)
                } else {
                  console.log('[CollectBlock]   Item entity no longer valid')
                }

                reject(new Error('Failed to pickup item'))
              }, 10000)
              tempEvents.subscribeTo(
                'entityGone',
                (entity: Entity) => {
                  if (entity === closest) {
                    clearTimeout(timeout)
                    tempEvents.cleanup()
                    // Diagnostic: Log successful pickup
                    console.log(`[CollectBlock] ✓ Item picked up: ${itemEntity.name || 'item'}`)
                    resolve()
                  }
                }
              )
            }
          )
          // PATCH (skillnet): dynamic=true. A static GoalFollow terminates on
          // arrival (goal_reached -> fullStop), so once the bot reaches the
          // drop's block it stops; in flowing water the drop keeps drifting and
          // the bot then idles until the 10s timeout, losing it. Dynamic keeps
          // the follow live so the bot chases the drifting drop until pickup.
          const followGoal = new goals.GoalFollow(closest as Entity, 0)
          bot.pathfinder.setGoal(followGoal, true)
          try {
            await waitForPickup
          } finally {
            // PATCH (skillnet): the executor never clears a dynamic goal (both
            // goal_reached clears are gated on !dynamicGoal) and the success
            // path never called setGoal(null), so the dead drop's GoalFollow
            // leaked out of collect() and later actions dragged the bot back
            // to the drop's last cell. Clear it iff it is still ours; the
            // identity guard spares a goal a canceller or new owner installed.
            if (bot.pathfinder.goal === followGoal) {
              try { bot.pathfinder.setGoal(null) } catch (e) {}
            }
          }
        } catch (err) {
          // @ts-expect-error
          console.log(err.stack)
          try {
            bot.pathfinder.setGoal(null)
          } catch (err) {}
          if (options.ignoreNoPath) {
            // @ts-expect-error
            if (err.message === 'Failed to pickup item') {
              bot.chat('Failed to pickup item! Skip it!')
            }
            break
          }
          throw err
        }
        break
      }
      default: {
        throw error(
          'UnknownType',
                    `Target ${closest.constructor.name} is not a Block or Entity!`
        )
      }
    }
    options.targets.removeTarget(closest)
  }
  bot.chat('Collect finish!')
}

const equipToolOptions = {
  requireHarvest: true,
  getFromChest: false,
  maxTools: 2
}

async function mineBlock (
  bot: Bot,
  block: Block,
  options: CollectOptionsFull
): Promise<void> {
  if (
    bot.blockAt(block.position)?.type !== block.type ||
        bot.blockAt(block.position)?.type === 0
  ) {
    options.targets.removeTarget(block)
    throw error('Invalid block', 'Block is not valid!')
    // @ts-expect-error
  } else if (!bot.pathfinder.movements.safeToBreak(block)) {
    options.targets.removeTarget(block)
    throw error('Unsafe block', 'Block is not safe to break!')
  }

  // Reject unbreakable blocks before attempting to mine
  if (block.hardness !== undefined && block.hardness < 0) {
    options.targets.removeTarget(block)
    throw error('Unbreakable',
      `Block ${block.name} at ${block.position} is unbreakable in survival mode (hardness=${block.hardness}). ` +
      `No tool can break this block.`)
  }

  await bot.tool.equipForBlock(block, equipToolOptions)

  if (!block.canHarvest((bot.heldItem != null) ? bot.heldItem.type : bot.heldItem)) {
    options.targets.removeTarget(block)
    throw error('NoItem', 'Bot does not have a harvestable tool!')
  }

  const tempEvents = new TemporarySubscriber(bot)
  // P2 fix: Increase distance threshold from 1.0 to 16.0
  // Problem: When mining logs from trees, items fall to ground far below the mined block.
  // E.g., mining log at y=68, item falls to y=64 → distance ≈ 4.6 blocks > 1.0
  // Solution: Allow 16 blocks to cover tall trees (~10 height) + horizontal drift
  tempEvents.subscribeTo('itemDrop', (entity: Entity) => {
    if (
      entity.position.distanceTo(block.position.offset(0.5, 0.5, 0.5)) <=
            16.0
    ) {
      options.targets.appendTarget(entity)
    }
  })
  // PATCH (skillnet): station-keeping for the dig + item-settle window.
  // collectblock has no motor control here (the approach goto already resolved
  // and set no goal), so during a multi-second dig flowing water washes the bot
  // out of interaction range; the block's drop then spawns far / drifts and
  // vanilla proximity pickup never fires. Hold the bot at the anchor (the
  // in-reach spot the approach left it) for the whole window so the block stays
  // reachable and the fresh drop is auto-collected on spawn. Engages only when
  // actually drifting (in water or measurably displaced) so dry-land behaviour
  // is byte-identical; yaw-only steer leaves the dig's pitch aim untouched.
  const anchor = bot.entity.position.clone()
  const stationKeep = (): void => {
    if (bot.entity == null) return
    const p = bot.entity.position
    const dx = anchor.x - p.x
    const dz = anchor.z - p.z
    const distSq = dx * dx + dz * dz
    const inWater = (bot.entity as any).isInWater === true // runtime flag, not in the Entity type
    if (!inWater && distSq <= 0.04) return // dry land, in place: do nothing
    if (distSq <= 0.04) {
      bot.setControlState('forward', false)
      bot.setControlState('jump', false)
      bot.setControlState('sneak', false)
      return
    }
    bot.look(Math.atan2(-dx, -dz), bot.entity.pitch, true)
    bot.setControlState('forward', true)
    if (inWater) {
      // directional swim: rise toward the anchor when below it, sink when above
      bot.setControlState('jump', anchor.y > p.y + 0.25)
      bot.setControlState('sneak', anchor.y < p.y - 1)
    }
  }
  bot.on('physicsTick', stationKeep)
  try {
    // @ts-expect-error
    await bot.dig(block)
    // Waiting for items to drop
    await new Promise<void>((resolve) => {
      let remainingTicks = 10
      tempEvents.subscribeTo('physicsTick', () => {
        remainingTicks--
        if (remainingTicks <= 0) {
          tempEvents.cleanup()
          resolve()
        }
      })
    })
  } finally {
    bot.removeListener('physicsTick', stationKeep)
    bot.setControlState('forward', false)
    bot.setControlState('jump', false)
    bot.setControlState('sneak', false)
    tempEvents.cleanup()
  }
}

/**
 * A set of options to apply when collecting the given targets.
 */
export interface CollectOptions {
  /**
     * If true, the target(s) will be appended to the existing target list instead of
     * starting a new task. Defaults to false.
     */
  append?: boolean

  /**
     * If true, errors will not be thrown when a path to the target block cannot
     * be found. The bot will attempt to choose the best available position it
     * can find, instead. Errors are still thrown if the bot cannot interact with
     * the block from it's final location. Defaults to false.
     */
  ignoreNoPath?: boolean

  /**
     * Gets the list of chest locations to use when storing items after the bot's
     * inventory becomes full. If undefined, it defaults to the chest location
     * list on the bot.collectBlock plugin.
     */
  chestLocations?: Vec3[]

  /**
     * When transferring items to a chest, this filter is used to determine what
     * items are allowed to be moved, and what items aren't allowed to be moved.
     * Defaults to the item filter specified on the bot.collectBlock plugin.
     */
  itemFilter?: ItemFilter

  /**
     * The total number of items to collect
     */
  count?: number
}

/**
 * A version of collect options where all values are assigned.
 */
interface CollectOptionsFull {
  append: boolean
  ignoreNoPath: boolean
  chestLocations: Vec3[]
  itemFilter: ItemFilter
  targets: Targets
  count: number
}

/**
 * The collect block plugin.
 */
export class CollectBlock {
  /**
     * The bot.
     */
  private readonly bot: Bot

  /**
     * The list of active targets being collected.
     */
  private readonly targets: Targets

  /**
     * The movements configuration to be sent to the pathfinder plugin.
     */
  movements?: Movements

  /**
     * A list of chest locations which the bot is allowed to empty their inventory into
     * if it becomes full while the bot is collecting resources.
     */
  chestLocations: Vec3[] = []

  /**
     * When collecting items, this filter is used to determine what items should be placed
     * into a chest if the bot's inventory becomes full. By default, returns true for all
     * items except for tools, weapons, and armor.
     *
     * @param item - The item stack in the bot's inventory to check.
     *
     * @returns True if the item should be moved into the chest. False otherwise.
     */
  itemFilter: ItemFilter = (item: Item) => {
    if (item.name.includes('helmet')) return false
    if (item.name.includes('chestplate')) return false
    if (item.name.includes('leggings')) return false
    if (item.name.includes('boots')) return false
    if (item.name.includes('shield')) return false
    if (item.name.includes('sword')) return false
    if (item.name.includes('pickaxe')) return false
    if (item.name.includes('axe')) return false
    if (item.name.includes('shovel')) return false
    if (item.name.includes('hoe')) return false
    return true
  }

  /**
     * Creates a new instance of the create block plugin.
     *
     * @param bot - The bot this plugin is acting on.
     */
  constructor (bot: Bot) {
    this.bot = bot
    this.targets = new Targets(bot)
    // @ts-expect-error
    this.movements = new Movements(bot, mcDataLoader(bot.version))
  }

  /**
     * If target is a block:
     * Causes the bot to break and collect the target block.
     *
     * If target is an item drop:
     * Causes the bot to collect the item drop.
     *
     * If target is an array containing items or blocks, preforms the correct action for
     * all targets in that array sorting dynamically by distance.
     *
     * @param target - The block(s) or item(s) to collect.
     * @param options - The set of options to use when handling these targets
     * @param cb - The callback that is called finished.
     */
  async collect (
    target: Collectable | Collectable[],
    options: CollectOptions | Callback = {},
    cb?: Callback
  ): Promise<void> {
    if (typeof options === 'function') {
      cb = options
      options = {}
    }
    // @ts-expect-error
    if (cb != null) return callbackify(this.collect)(target, options, cb)

    const optionsFull: CollectOptionsFull = {
      append: options.append ?? false,
      ignoreNoPath: options.ignoreNoPath ?? false,
      chestLocations: options.chestLocations ?? this.chestLocations,
      itemFilter: options.itemFilter ?? this.itemFilter,
      targets: this.targets,
      count: options.count ?? Infinity
    }

    if (this.bot.pathfinder == null) {
      throw error(
        'UnresolvedDependency',
        'The mineflayer-collectblock plugin relies on the mineflayer-pathfinder plugin to run!'
      )
    }

    if (this.bot.tool == null) {
      throw error(
        'UnresolvedDependency',
        'The mineflayer-collectblock plugin relies on the mineflayer-tool plugin to run!'
      )
    }

    if (this.movements != null) {
      this.bot.pathfinder.setMovements(this.movements)
    }

    if (!optionsFull.append) {
      await this.cancelTask()
      // PATCH (skillnet): a fresh (non-append) task starts un-cancelled. The
      // reset stays inside the !append branch: an append joins a possibly
      // cancelling task and must not resurrect it.
      this.targets.cancelled = false
    }
    if (Array.isArray(target)) {
      this.targets.appendTargets(target)
    } else {
      this.targets.appendTarget(target)
    }

    try {
      await collectAll(this.bot, optionsFull)
      this.targets.clear()
    } catch (err) {
      this.targets.clear()
      // Ignore path stopped error for cancelTask to work properly (imo we shouldn't throw any pathing errors)
      // PATCH (skillnet): GoalChanged is the other cancellation-shaped
      // rejection (the canceller cleared or replaced the goal); swallow it the
      // same way so a cancelled collect resolves instead of rejecting.
      // @ts-expect-error
      if (err.name !== 'PathStopped' && err.name !== 'GoalChanged') throw err
    } finally {
      // @ts-expect-error
      this.bot.emit('collectBlock_finished')
    }
  }

  /**
     * Loads all touching blocks of the same type to the given block and returns them as an array.
     * This effectively acts as a flood fill algorithm to retrieve blocks in the same ore vein and similar.
     *
     * @param block - The starting block.
     * @param maxBlocks - The maximum number of blocks to look for before stopping.
     * @param maxDistance - The max distance from the starting block to look.
     * @param floodRadius - The max distance distance from block A to block B to be considered "touching"
     */
  findFromVein (
    block: Block,
    maxBlocks = 100,
    maxDistance = 16,
    floodRadius = 1
  ): Block[] {
    // @ts-expect-error
    return findFromVein(
      this.bot,
      // @ts-expect-error
      block,
      maxBlocks,
      maxDistance,
      floodRadius
    )
  }

  /**
     * Cancels the current collection task, if still active.
     *
     * @param cb - The callback to use when the task is stopped.
     */
  async cancelTask (cb?: Callback): Promise<void> {
    if (this.targets.empty) {
      if (cb != null) cb()
      return await Promise.resolve()
    }
    // PATCH (skillnet): make cancellation cooperative, not just a path stop.
    // With ignoreNoPath the loop used to swallow the PathStopped and walk the
    // remaining target list to exhaustion before 'collectBlock_finished'.
    this.targets.cancelled = true
    this.bot.pathfinder.stop()
    if (cb != null) {
      // @ts-expect-error
      this.bot.once('collectBlock_finished', cb)
    }
    await once(this.bot, 'collectBlock_finished')
  }
}
