"""
Primitive Knowledge - Combat
killMob primitive knowledge

Migrated from primitive_knowledge.py
All fix_hint, fix_strategy, check_hint fields removed
"""

from typing import List
from skillnet.core.knowledge_base import KnowledgeItem, KnowledgeDomain, KnowledgeCategory


# ============================================================
# killMob Primitive Knowledge
# ============================================================

COMBAT_KNOWLEDGE: List[KnowledgeItem] = [
    # Preconditions
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="killMob_target_exists",
        fact=(
            "killMob requires the target mob to exist and be visible. "
            "Uses bot.nearestEntity() to find mobs by name. "
            "Mob must be within render distance and in loaded chunks."
        ),
        keywords=["killmob", "target", "entity", "exists", "visible", "find"],
        conditions=["calling killMob"],
        implications=[
            "mob must be nearby",
            "in loaded chunks only",
            "may need to explore first",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="killMob_attack_range",
        fact=(
            "killMob requires being within 3 blocks for melee attacks. "
            "Bot pathfinds to mob and maintains attack distance. "
            "Ranged weapons (bow) have unlimited range but need line of sight."
        ),
        keywords=["killmob", "range", "distance", "melee", "attack", "reach"],
        conditions=["attacking mobs"],
        implications=[
            "melee needs close range",
            "bot will pathfind to mob",
            "bow for ranged combat",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="killMob_weapon_equipped",
        fact=(
            "killMob is more effective with a weapon equipped. "
            "Sword deals most damage per hit, axe deals critical damage. "
            "Bare fists deal minimal damage (1 heart)."
        ),
        keywords=["killmob", "weapon", "sword", "axe", "damage", "equip"],
        conditions=["combat preparation"],
        implications=[
            "weapon recommended",
            "sword is fastest",
            "axe for crits",
        ],
    ),

    # Failure patterns
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="killMob_entity_not_found",
        fact=(
            "killMob fails with 'Entity not found' or 'No X nearby' when target doesn't exist. "
            "Mob may have despawned (>128 blocks away) or hasn't spawned yet. "
            "Some mobs only spawn in specific conditions (night, biome, dimension)."
        ),
        keywords=["killmob", "entity", "not found", "nearby", "despawn", "spawn"],
        conditions=["mob not present"],
        implications=[
            "mob may have despawned",
            "check spawn conditions",
            "explore to find mobs",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="killMob_out_of_range",
        fact=(
            "killMob fails with 'Out of range' or 'Too far' when mob is beyond attack distance. "
            "Melee range is 3 blocks. Mob may be moving or pathfinding may fail."
        ),
        keywords=["killmob", "out of range", "too far", "reach", "distance"],
        conditions=["mob out of attack range"],
        implications=[
            "pathfind closer to mob",
            "mob may be moving",
            "follow and retry",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="killMob_mob_escaped",
        fact=(
            "killMob may fail if mob escapes during combat. "
            "Fast mobs can outrun the bot. Flying mobs may go out of reach. "
            "Water mobs may dive too deep."
        ),
        keywords=["killmob", "escape", "fast", "flying", "water", "chase"],
        conditions=["mob escapes"],
        implications=[
            "some mobs are fast",
            "flying mobs hard to reach",
            "may need ranged weapon",
        ],
    ),

    # Effects
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="killMob_drops",
        fact=(
            "Killed mobs drop items and experience. "
            "Drops vary by mob type: cows drop beef and leather, zombies drop rotten flesh. "
            "Dropped items must be collected before despawning (5 minutes)."
        ),
        keywords=["killmob", "drops", "items", "loot", "experience", "xp"],
        conditions=["after killing mob"],
        implications=[
            "items drop on ground",
            "collect before despawn",
            "different mobs different loot",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="killMob_attack_cooldown",
        fact=(
            "Weapons have attack cooldown for full damage. "
            "Sword cooldown is 0.625 seconds, axe is 1 second. "
            "Attacking before cooldown deals reduced damage."
        ),
        keywords=["killmob", "cooldown", "attack", "speed", "damage", "wait"],
        conditions=["combat timing"],
        implications=[
            "wait for full cooldown",
            "sword is faster",
            "rapid clicks = less damage",
        ],
    ),

    # ============================================================
    # Mob-Specific Combat Behavior (1.19+ Knowledge)
    # ============================================================

    # Creeper
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="creeper_combat_behavior",
        fact=(
            "Creepers explode when close to player (1.5 blocks) after 1.5 second fuse. "
            "Killing before explosion prevents damage. Silent until fuse starts (hissing sound). "
            "Explosion destroys nearby blocks and items."
        ),
        keywords=["creeper", "explode", "fuse", "hiss", "explosion", "blocks"],
        conditions=["fighting creepers"],
        implications=[
            "hit and retreat strategy",
            "don't let them get close",
            "listen for hissing",
            "can destroy your items",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="creeper_flee_from_cats",
        fact=(
            "Creepers flee from cats and ocelots within 6 blocks. "
            "Charged creepers (lightning-struck) have 2x explosion power and drop mob heads. "
            "Skeleton arrows can make creepers drop music discs if skeleton kills creeper."
        ),
        keywords=["creeper", "cat", "ocelot", "flee", "charged", "lightning", "music disc"],
        conditions=["creeper nearby", "need mob heads", "need music disc"],
        implications=[
            "use cats for creeper protection",
            "charged creepers for mob heads",
            "skeleton + creeper = music disc",
        ],
    ),

    # Skeleton
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="skeleton_combat_behavior",
        fact=(
            "Skeletons shoot arrows from up to 15 blocks away. "
            "They strafe (move sideways) while shooting. "
            "Shields block arrow damage completely. Close distance quickly to force melee."
        ),
        keywords=["skeleton", "arrow", "ranged", "strafe", "shield", "bow"],
        conditions=["fighting skeletons"],
        implications=[
            "close gap quickly",
            "use cover to approach",
            "shield blocks arrows",
            "melee when close",
        ],
    ),

    # Zombie
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="zombie_combat_behavior",
        fact=(
            "Zombies are slow melee attackers with 20 HP. "
            "Can break wooden doors on hard difficulty. "
            "Baby zombies are faster, smaller, and fit through 1-block gaps."
        ),
        keywords=["zombie", "melee", "slow", "door", "baby", "small"],
        conditions=["fighting zombies"],
        implications=[
            "easy to kite (hit and run)",
            "can spawn with armor",
            "baby zombies more dangerous",
            "burn in sunlight",
        ],
    ),

    # Spider
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="spider_combat_behavior",
        fact=(
            "Spiders can climb walls and are neutral during daytime. "
            "2 blocks wide but can fit through 1-block gaps. "
            "Cave spiders (mineshafts) apply poison effect on attack."
        ),
        keywords=["spider", "climb", "neutral", "day", "poison", "cave spider", "wall"],
        conditions=["fighting spiders"],
        implications=[
            "walls don't stop them",
            "safe during day unless provoked",
            "cave spiders are poisonous",
            "can ambush from above",
        ],
    ),

    # Enderman
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="enderman_combat_behavior",
        fact=(
            "Endermen teleport when damaged or hit by projectiles. "
            "Take damage from water and rain. "
            "Become hostile when looked at (crosshair on upper body). "
            "Look at their legs to be near without aggro."
        ),
        keywords=["enderman", "teleport", "water", "eye contact", "aggro", "rain"],
        conditions=["fighting endermen", "avoiding enderman aggro"],
        implications=[
            "use water for damage",
            "trap in 2-block high space to prevent teleport",
            "avoid looking at face",
            "rain damages them",
        ],
    ),

    # Mob despawn mechanics
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.FAILURE,
        name="combat_mob_despawn",
        fact=(
            "Hostile mobs despawn instantly if player is >128 blocks away. "
            "Mobs can despawn randomly if no player within 32 blocks for extended time. "
            "Named mobs (name tag) and mobs holding items never despawn."
        ),
        keywords=["despawn", "distance", "named", "mob", "disappear"],
        conditions=["mob disappeared unexpectedly"],
        implications=[
            "stay within 128 blocks of target",
            "name tags prevent despawn",
            "mobs holding items persist",
        ],
    ),

    # Group spawning
    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="combat_group_spawning",
        fact=(
            "Mobs often spawn in groups (1-4 for most hostile mobs). "
            "Zombie reinforcements can spawn when attacking zombies (up to 50 blocks away). "
            "Spider jockeys (skeleton riding spider) are rare but dangerous."
        ),
        keywords=["group", "spawn", "reinforcement", "jockey", "multiple"],
        conditions=["encountering multiple mobs"],
        implications=[
            "prioritize dangerous threats first",
            "avoid fighting in open areas",
            "zombies can call reinforcements",
        ],
    ),

    # ============================================================
    # 1.19 Wild Update - Warden (Most Dangerous Mob)
    # ============================================================

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="warden_combat_behavior",
        fact=(
            "Warden is BLIND but detects vibrations and can smell players. "
            "Has highest HP (500) and melee damage (30 = 15 hearts) in game. "
            "Sonic boom ranged attack ignores armor, deals 10 damage (5 hearts), 15 block range. "
            "DO NOT FIGHT - sneak to evade. Fastest mob when angry."
        ),
        keywords=["warden", "blind", "vibration", "sonic boom", "damage", "deep dark", "sneak"],
        conditions=["encountering warden", "exploring deep dark"],
        implications=[
            "DO NOT fight directly",
            "sneak to avoid detection",
            "use distractions (throw items)",
            "run if detected",
            "sonic boom ignores armor",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.PRECONDITION,
        name="warden_spawn_mechanics",
        fact=(
            "Warden spawns after sculk shrieker activates 4 times within 10 minutes. "
            "Warning levels reset after 10 minutes of no shrieking. "
            "Breaking shrieker with Silk Touch doesn't trigger activation. "
            "Darkness effect applied when warden nearby (screen dims)."
        ),
        keywords=["warden", "spawn", "shrieker", "warning", "darkness", "sculk", "silk touch"],
        conditions=["approaching sculk shrieker", "in deep dark"],
        implications=[
            "break shriekers with silk touch",
            "wait 10 min between activations",
            "darkness effect = warden is near",
            "4 shrieks = warden spawns",
        ],
    ),

    KnowledgeItem(
        domain=KnowledgeDomain.PRIMITIVE,
        category=KnowledgeCategory.EFFECT,
        name="warden_distraction_tactics",
        fact=(
            "Warden can be distracted by throwing items (snowballs, eggs, arrows). "
            "Warden investigates thrown projectiles (vibration source). "
            "Sneak-walking does not create vibrations Warden can detect. "
            "Running, breaking blocks, opening chests all create detectable vibrations."
        ),
        keywords=["warden", "distract", "throw", "sneak", "vibration", "projectile"],
        conditions=["evading warden"],
        implications=[
            "throw items to distract",
            "sneak-walk to avoid detection",
            "don't run or break blocks",
            "wait for warden to investigate distraction",
        ],
    ),
]


def get_all_combat_knowledge() -> List[KnowledgeItem]:
    """Get all combat primitive knowledge items."""
    return COMBAT_KNOWLEDGE.copy()
