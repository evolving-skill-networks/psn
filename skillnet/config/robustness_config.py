"""
Configuration switches for robustness-enhancement features.
Used for A/B testing and gradual rollout.
"""


class RobustnessConfig:
    """Robustness-enhancement configuration"""

    # === Parameter-extraction enhancements (Part 1) ===
    # Whether to enable JSON-mode constraints
    ENABLE_JSON_MODE = True

    # Whether to use robust JSON extraction (extract_json_from_response)
    ENABLE_ROBUST_JSON_PARSING = True

    # Whether to enable few-shot examples
    ENABLE_PARAMETER_FEWSHOT = True

    # Whether to enable the simplified LLM fallback
    ENABLE_SIMPLIFIED_FALLBACK = True

    # Whether to enable the regex emergency fallback
    ENABLE_REGEX_EMERGENCY = True

    # === EffectMatcher enhancements (Part 2) ===
    # Whether to enable the enhanced LLM prompt (CRITICAL RULES + examples)
    ENABLE_ENHANCED_EFFECT_PROMPT = True

    # Whether to enable response validation
    ENABLE_EFFECT_VALIDATION = True

    # Whether to enable the extended rule-extraction patterns
    ENABLE_EXTENDED_RULE_PATTERNS = True

    # === Skill lifecycle management (Part 3) ===
    # Whether to enable the delayed-deletion mechanism
    ENABLE_DELAYED_DELETION = True

    # Failure threshold for delayed deletion (after how many failures to delete)
    DELETION_FAILURE_THRESHOLD = 3

    # Whether to enable auto-persist on success
    ENABLE_AUTO_PERSIST = True

    # === Milestone semantic consistency (Part 4) ===
    # Whether to allow wood to count as fuel (matches Minecraft game logic)
    ENABLE_WOOD_AS_FUEL = True

    # Whether to enable system-level milestone validation (defensive check)
    ENABLE_MILESTONE_VALIDATION = True

    # === Code-generation enhancements (Part 5) ===
    # Whether to enable auto-reparameterization (auto-add parameters to non-parameterized functions for weaker models)
    ENABLE_AUTO_REPARAMETERIZE = True

    # === Critic-evaluation enhancements (Part 6) ===
    # Whether to enable the enhanced Critic prompt (DELTA-semantic reinforcement rules for weaker models)
    ENABLE_ENHANCED_CRITIC_PROMPT = True

    # === Global switches ===
    @classmethod
    def enable_all(cls):
        """Enable all enhancements (used for weaker models)"""
        cls.ENABLE_JSON_MODE = True
        cls.ENABLE_ROBUST_JSON_PARSING = True
        cls.ENABLE_PARAMETER_FEWSHOT = True
        cls.ENABLE_SIMPLIFIED_FALLBACK = True
        cls.ENABLE_REGEX_EMERGENCY = True
        cls.ENABLE_ENHANCED_EFFECT_PROMPT = True
        cls.ENABLE_EFFECT_VALIDATION = True
        cls.ENABLE_EXTENDED_RULE_PATTERNS = True
        cls.ENABLE_DELAYED_DELETION = True
        cls.ENABLE_AUTO_PERSIST = True
        cls.ENABLE_WOOD_AS_FUEL = True
        cls.ENABLE_MILESTONE_VALIDATION = True
        cls.ENABLE_AUTO_REPARAMETERIZE = True
        cls.ENABLE_ENHANCED_CRITIC_PROMPT = True

    @classmethod
    def disable_all_enhancements(cls):
        """Disable all enhancements (used for baseline tests)"""
        cls.ENABLE_JSON_MODE = False
        cls.ENABLE_ROBUST_JSON_PARSING = False
        cls.ENABLE_PARAMETER_FEWSHOT = False
        cls.ENABLE_SIMPLIFIED_FALLBACK = False
        cls.ENABLE_REGEX_EMERGENCY = False
        cls.ENABLE_ENHANCED_EFFECT_PROMPT = False
        cls.ENABLE_EFFECT_VALIDATION = False
        cls.ENABLE_EXTENDED_RULE_PATTERNS = False
        cls.ENABLE_DELAYED_DELETION = False
        cls.ENABLE_AUTO_PERSIST = False
        cls.ENABLE_WOOD_AS_FUEL = False
        cls.ENABLE_MILESTONE_VALIDATION = False
        cls.ENABLE_AUTO_REPARAMETERIZE = False
        cls.ENABLE_ENHANCED_CRITIC_PROMPT = False

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        import os
        cls.ENABLE_JSON_MODE = os.getenv("ENABLE_JSON_MODE", "true").lower() == "true"
        cls.ENABLE_ROBUST_JSON_PARSING = os.getenv("ENABLE_ROBUST_JSON_PARSING", "true").lower() == "true"
        cls.ENABLE_PARAMETER_FEWSHOT = os.getenv("ENABLE_PARAMETER_FEWSHOT", "true").lower() == "true"
        cls.ENABLE_SIMPLIFIED_FALLBACK = os.getenv("ENABLE_SIMPLIFIED_FALLBACK", "true").lower() == "true"
        cls.ENABLE_REGEX_EMERGENCY = os.getenv("ENABLE_REGEX_EMERGENCY", "true").lower() == "true"
        cls.ENABLE_ENHANCED_EFFECT_PROMPT = os.getenv("ENABLE_ENHANCED_EFFECT_PROMPT", "true").lower() == "true"
        cls.ENABLE_EFFECT_VALIDATION = os.getenv("ENABLE_EFFECT_VALIDATION", "true").lower() == "true"
        cls.ENABLE_EXTENDED_RULE_PATTERNS = os.getenv("ENABLE_EXTENDED_RULE_PATTERNS", "true").lower() == "true"
        cls.ENABLE_DELAYED_DELETION = os.getenv("ENABLE_DELAYED_DELETION", "true").lower() == "true"
        cls.DELETION_FAILURE_THRESHOLD = int(os.getenv("DELETION_FAILURE_THRESHOLD", "3"))
        cls.ENABLE_AUTO_PERSIST = os.getenv("ENABLE_AUTO_PERSIST", "true").lower() == "true"
        cls.ENABLE_WOOD_AS_FUEL = os.getenv("ENABLE_WOOD_AS_FUEL", "true").lower() == "true"
        cls.ENABLE_MILESTONE_VALIDATION = os.getenv("ENABLE_MILESTONE_VALIDATION", "true").lower() == "true"
        cls.ENABLE_AUTO_REPARAMETERIZE = os.getenv("ENABLE_AUTO_REPARAMETERIZE", "true").lower() == "true"
        cls.ENABLE_ENHANCED_CRITIC_PROMPT = os.getenv("ENABLE_ENHANCED_CRITIC_PROMPT", "true").lower() == "true"
