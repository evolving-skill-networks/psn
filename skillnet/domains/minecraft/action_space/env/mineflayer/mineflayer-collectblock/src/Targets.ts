import { Bot } from 'mineflayer'
import { Block } from 'prismarine-block'
import { Entity } from 'prismarine-entity'

export type Collectable = Block | Entity

export class Targets {
  private readonly bot: Bot
  private targets: Collectable[] = []

  constructor (bot: Bot) {
    this.bot = bot
  }

  appendTargets (targets: Collectable[]): void {
    for (const target of targets) {
      this.appendTarget(target)
    }
  }

  appendTarget (target: Collectable): void {
    if (this.targets.includes(target)) return
    this.targets.push(target)
  }

  /**
   * Gets the closest REACHABLE target, prioritizing same-Y blocks over far
   * vertical (in either direction). Vertical traversal in Minecraft costs more
   * than horizontal of the same Euclidean distance — going UP requires placing
   * blocks or jumping, going DOWN requires digging or falling.
   *
   * Score = horizontal_distance + vertical_penalty(|Δy|)
   *
   * The previous implementation gave penalty=0 for ANY block at-or-below bot
   * (motivated by tree-bottom-first harvesting). That over-prioritized blocks
   * far below bot — e.g. a target 10 blocks down would beat one 5 blocks
   * horizontal. This symmetric design preserves tree harvesting (bottom log at
   * Δy=0 still wins via penalty=0) while penalising far-vertical targets in
   * either direction. Ascent is penalised more heavily than descent because
   * placing build-up blocks is slower than digging-down.
   *
   * @returns The closest reachable target, or null if there are no targets.
   */
  getClosest (): Collectable | null {
    // P0: defensive check - bot.entity may be undefined on death/disconnect
    if (!this.bot.entity) {
      console.warn('[Targets.getClosest] bot.entity is undefined, returning null')
      return null
    }

    const botPos = this.bot.entity.position
    const botY = botPos.y

    let best: Collectable | null = null
    let bestScore: number = Infinity

    for (const target of this.targets) {
      const targetY = target.position.y
      const horizontalDist = Math.sqrt(
        Math.pow(target.position.x - botPos.x, 2) +
        Math.pow(target.position.z - botPos.z, 2)
      )
      const verticalDist = targetY - botY
      const absDy = Math.abs(verticalDist)

      // Symmetric vertical penalty with ascent/descent asymmetry.
      let reachabilityPenalty: number
      if (absDy === 0) {
        // Same height — best (tree bottom log wins via this branch)
        reachabilityPenalty = 0
      } else if (absDy <= 2) {
        // ±1-2 blocks: reachable by jumping (up) / walking or falling (down). Up slightly more expensive.
        reachabilityPenalty = verticalDist > 0 ? absDy * 2 : absDy * 1.5
      } else {
        // ≥3 blocks: long vertical distance, both expensive. Up costs more (needs block placement to bridge);
        // Down moderate (needs digging down / staircase).
        reachabilityPenalty = verticalDist > 0 ? absDy * 10 : absDy * 5
      }

      const score = horizontalDist + reachabilityPenalty

      if (score < bestScore) {
        bestScore = score
        best = target
      }
    }

    return best
  }

  get empty (): boolean {
    return this.targets.length === 0
  }

  clear (): void {
    this.targets.length = 0
  }

  removeTarget (target: Collectable): void {
    const index = this.targets.indexOf(target)
    if (index < 0) return
    this.targets.splice(index, 1)
  }
}
