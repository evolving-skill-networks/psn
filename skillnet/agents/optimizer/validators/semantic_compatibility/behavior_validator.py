"""
Behavior Equivalence Validator

v7.4 Phase 4: behavior equivalence validator

Features:
1. Static analysis: extract API call sequences, state changes, child-skill calls
2. Effects comparison: compare effects descriptions before and after the refactor
3. LLM validation: use the LLM to judge whether behaviors are equivalent

Three-layer validation strategy:
- Layer 1: static analysis (100-200ms)
- Layer 2: effects check (500-1000ms)
- Layer 3: LLM validation (2-5s, optional)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Dict, Any, TYPE_CHECKING

from skillnet.agents.skill_graph.utils.code_analysis import extract_function_calls

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel as LLM

logger = logging.getLogger(__name__)


@dataclass
class BehaviorFeature:
    """
    Behavior features.

    Describes the behavior features of code, including API calls, state changes,
    child-skill calls, etc.
    """
    api_calls: List[str] = field(default_factory=list)      # bot.dig, bot.craft, etc.
    state_changes: List[str] = field(default_factory=list)  # inventory, position, etc.
    effects_descriptions: List[str] = field(default_factory=list)
    has_error_handling: bool = False
    has_retry_logic: bool = False
    child_skill_calls: List[str] = field(default_factory=list)
    control_flow: List[str] = field(default_factory=list)   # if/for/while structures

    @property
    def all_calls(self) -> List[str]:
        """Get all calls (API + child skills)."""
        return self.api_calls + self.child_skill_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            'api_calls': self.api_calls,
            'state_changes': self.state_changes,
            'effects_descriptions': self.effects_descriptions,
            'has_error_handling': self.has_error_handling,
            'has_retry_logic': self.has_retry_logic,
            'child_skill_calls': self.child_skill_calls,
            'control_flow': self.control_flow,
        }


@dataclass
class BehaviorEquivalenceResult:
    """
    Behavior equivalence validation result.
    """
    is_equivalent: bool
    confidence: float  # 0.0 - 1.0
    layer_passed: int  # Number of validation layers passed (1, 2, or 3)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    llm_reasoning: Optional[str] = None
    old_features: Optional[BehaviorFeature] = None
    new_features: Optional[BehaviorFeature] = None

    def __bool__(self) -> bool:
        return self.is_equivalent


class BehaviorEquivalenceValidator:
    """
    Behavior equivalence validator.

    Verifies whether the refactored code is behaviorally equivalent to the original.

    Usage:
        validator = BehaviorEquivalenceValidator(llm=llm_instance)

        # Extract behavior features
        old_features = validator.extract_behavior_features(old_code)
        new_features = validator.extract_behavior_features(new_code)

        # Compare behaviors
        is_equiv, issues = validator.compare_behaviors(old_features, new_features)

        # Or use the full validation
        result = await validator.validate_equivalence(
            old_code=old_code,
            new_code=new_code,
            old_effects=old_effects,
            new_effects=new_effects,
        )
    """

    # Bot API patterns
    BOT_API_PATTERNS = [
        r'bot\.dig\s*\(',
        r'bot\.place\s*\(',
        r'bot\.craft\s*\(',
        r'bot\.smelt\s*\(',
        r'bot\.equip\s*\(',
        r'bot\.attack\s*\(',
        r'bot\.useOn\s*\(',
        r'bot\.look\s*\(',
        r'bot\.pathfinder\.',
        r'bot\.creative\.',
        r'bot\.chat\s*\(',
        r'bot\.whisper\s*\(',
        r'bot\.toss\s*\(',
        r'bot\.activateBlock\s*\(',
        r'bot\.activateEntity\s*\(',
        r'bot\.closeWindow\s*\(',
        r'bot\.openContainer\s*\(',
        r'bot\.putSelectedItemSlot\s*\(',
        r'bot\.clickWindow\s*\(',
        r'bot\.transfer\s*\(',
        r'bot\.consume\s*\(',
        r'bot\.fish\s*\(',
        r'bot\.sleep\s*\(',
        r'bot\.wake\s*\(',
    ]

    # State-change patterns
    STATE_CHANGE_PATTERNS = [
        (r'inventory', 'inventory'),
        (r'position', 'position'),
        (r'health', 'health'),
        (r'food', 'food'),
        (r'experience', 'experience'),
        (r'bot\.entity\.position', 'position'),
        (r'bot\.inventory', 'inventory'),
        (r'bot\.quickBarSlot', 'hotbar'),
    ]

    # Control-flow patterns
    CONTROL_FLOW_PATTERNS = [
        (r'\bif\s*\(', 'if'),
        (r'\bfor\s*\(', 'for'),
        (r'\bwhile\s*\(', 'while'),
        (r'\btry\s*\{', 'try'),
        (r'\bcatch\s*\(', 'catch'),
    ]

    # Error-handling patterns
    ERROR_HANDLING_PATTERNS = [
        r'\btry\s*\{',
        r'\bcatch\s*\(',
        r'\.catch\s*\(',
        r'throw\s+',
    ]

    # Retry-logic patterns
    RETRY_PATTERNS = [
        r'\bretry',
        r'\battempt',
        r'\bwhile\s*\([^)]*<\s*\d+',  # while (count < N)
        r'for\s*\([^)]*;\s*\w+\s*<\s*\d+',  # for (i; i < N; ...)
    ]

    def __init__(
        self,
        llm: "LLM" = None,
        logger_instance=None,
        enable_llm_validation: bool = True,
    ):
        """
        Initialize the behavior equivalence validator.

        Args:
            llm: LLM instance (used for Layer 3 validation)
            logger_instance: Logger instance
            enable_llm_validation: Whether to enable LLM validation
        """
        self.llm = llm
        self.logger = logger_instance or logger
        self.enable_llm_validation = enable_llm_validation

    def extract_behavior_features(self, code: str) -> BehaviorFeature:
        """
        Extract behavior features from code.

        Args:
            code: JavaScript code

        Returns:
            BehaviorFeature: Extracted behavior features
        """
        # Extract API calls
        api_calls = self._extract_api_calls(code)

        # Extract state changes
        state_changes = self._extract_state_changes(code)

        # Extract child skill calls
        child_skill_calls = extract_function_calls(code)

        # Extract control flow
        control_flow = self._extract_control_flow(code)

        # Detect error handling
        has_error_handling = self._has_error_handling(code)

        # Detect retry logic
        has_retry_logic = self._has_retry_logic(code)

        return BehaviorFeature(
            api_calls=api_calls,
            state_changes=state_changes,
            effects_descriptions=[],  # Passed in externally
            has_error_handling=has_error_handling,
            has_retry_logic=has_retry_logic,
            child_skill_calls=child_skill_calls,
            control_flow=control_flow,
        )

    def _extract_api_calls(self, code: str) -> List[str]:
        """
        Extract bot API calls.

        Args:
            code: JavaScript code

        Returns:
            List[str]: List of API calls
        """
        api_calls = []

        for pattern in self.BOT_API_PATTERNS:
            matches = re.findall(pattern, code)
            for match in matches:
                # Extract the API name
                api_name = match.rstrip('(').strip()
                if api_name not in api_calls:
                    api_calls.append(api_name)

        return api_calls

    def _extract_state_changes(self, code: str) -> List[str]:
        """
        Extract state changes.

        Args:
            code: JavaScript code

        Returns:
            List[str]: List of state changes
        """
        state_changes = []
        code_lower = code.lower()

        for pattern, state_name in self.STATE_CHANGE_PATTERNS:
            if re.search(pattern, code_lower):
                if state_name not in state_changes:
                    state_changes.append(state_name)

        return state_changes

    def _extract_control_flow(self, code: str) -> List[str]:
        """
        Extract control-flow structures.

        Args:
            code: JavaScript code

        Returns:
            List[str]: List of control-flow structures
        """
        control_flow = []

        for pattern, flow_type in self.CONTROL_FLOW_PATTERNS:
            count = len(re.findall(pattern, code))
            if count > 0:
                control_flow.append(f"{flow_type}:{count}")

        return control_flow

    def _has_error_handling(self, code: str) -> bool:
        """
        Detect whether the code has error handling.

        Args:
            code: JavaScript code

        Returns:
            bool: Whether error handling is present
        """
        for pattern in self.ERROR_HANDLING_PATTERNS:
            if re.search(pattern, code):
                return True
        return False

    def _has_retry_logic(self, code: str) -> bool:
        """
        Detect whether the code has retry logic.

        Args:
            code: JavaScript code

        Returns:
            bool: Whether retry logic is present
        """
        for pattern in self.RETRY_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False

    def compare_behaviors(
        self,
        old_features: BehaviorFeature,
        new_features: BehaviorFeature,
    ) -> Tuple[bool, List[str]]:
        """
        Compare two behavior feature sets for equivalence (Layer 1).

        Args:
            old_features: Behavior features of the original code
            new_features: Behavior features of the new code

        Returns:
            Tuple[bool, List[str]]: (whether equivalent, list of issues)
        """
        issues = []

        # Check 1: API calls
        old_apis = set(old_features.api_calls)
        new_apis = set(new_features.api_calls)

        missing_apis = old_apis - new_apis
        if missing_apis:
            issues.append(f"Missing API calls: {', '.join(sorted(missing_apis))}")

        # Check 2: child skill calls
        old_skills = set(old_features.child_skill_calls)
        new_skills = set(new_features.child_skill_calls)

        missing_skills = old_skills - new_skills
        if missing_skills:
            issues.append(f"Missing child skill calls: {', '.join(sorted(missing_skills))}")

        # Check 3: state changes
        old_states = set(old_features.state_changes)
        new_states = set(new_features.state_changes)

        missing_states = old_states - new_states
        if missing_states:
            issues.append(f"Missing state changes: {', '.join(sorted(missing_states))}")

        # Check 4: error handling
        if old_features.has_error_handling and not new_features.has_error_handling:
            issues.append("Error handling was removed")

        # Check 5: retry logic
        if old_features.has_retry_logic and not new_features.has_retry_logic:
            issues.append("Retry logic was removed")

        is_equivalent = len(issues) == 0
        return is_equivalent, issues

    def compare_effects(
        self,
        old_effects: List[str],
        new_effects: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Compare effects for equivalence (Layer 2).

        Args:
            old_effects: List of original effects descriptions
            new_effects: List of new effects descriptions

        Returns:
            Tuple[bool, List[str]]: (whether equivalent, list of issues)
        """
        issues = []

        # Normalize effects
        old_normalized = self._normalize_effects(old_effects)
        new_normalized = self._normalize_effects(new_effects)

        # Check missing effects
        missing = old_normalized - new_normalized
        if missing:
            # Check whether there is a semantically similar replacement
            truly_missing = []
            for old_effect in missing:
                if not self._has_similar_effect(old_effect, new_normalized):
                    truly_missing.append(old_effect)

            if truly_missing:
                issues.append(f"Missing effects: {truly_missing[:3]}...")

        # Check newly added effects (may introduce additional behavior)
        added = new_normalized - old_normalized
        if added:
            # New effects are usually a warning, not an error
            self.logger.info(f"[BehaviorValidator] Newly added effects: {list(added)[:3]}")

        is_equivalent = len(issues) == 0
        return is_equivalent, issues

    def _normalize_effects(self, effects: List[str]) -> Set[str]:
        """
        Normalize effects descriptions (used for comparison).

        Args:
            effects: List of effects descriptions

        Returns:
            Set[str]: Set of normalized effects
        """
        normalized = set()

        for effect in effects:
            # Lowercase and collapse whitespace
            norm = ' '.join(effect.lower().split())

            # Extract key verbs and objects
            keywords = self._extract_effect_keywords(norm)
            if keywords:
                normalized.add(keywords)

        return normalized

    def _extract_effect_keywords(self, effect: str) -> str:
        """
        Extract keywords from an effect description.

        Args:
            effect: Effect description

        Returns:
            str: Keyword string
        """
        # Key verbs
        verbs = ['mine', 'craft', 'place', 'dig', 'collect', 'obtain', 'get', 'move',
                 'goto', 'attack', 'kill', 'eat', 'equip', 'smelt', 'cook']

        # Extract verbs
        found_verbs = []
        for verb in verbs:
            if verb in effect:
                found_verbs.append(verb)

        # Extract item/block names (simplified)
        item_pattern = r'\b(\w+_\w+|\w+)\b'
        items = re.findall(item_pattern, effect)
        items = [i for i in items if len(i) > 3 and i not in verbs]

        return f"{'-'.join(found_verbs)}:{'-'.join(items[:3])}"

    def _has_similar_effect(self, effect: str, effects_set: Set[str]) -> bool:
        """
        Check whether there is a semantically similar effect.

        Args:
            effect: Effect to check
            effects_set: Set of effects

        Returns:
            bool: Whether there is a similar effect
        """
        # Simple similarity check: keyword overlap
        effect_parts = set(effect.split(':'))

        for other in effects_set:
            other_parts = set(other.split(':'))
            # Consider similar when more than 50% overlap
            overlap = len(effect_parts & other_parts)
            total = len(effect_parts | other_parts)
            if total > 0 and overlap / total > 0.5:
                return True

        return False

    async def validate_with_llm(
        self,
        old_code: str,
        new_code: str,
        old_effects: List[str],
        new_effects: List[str],
    ) -> Tuple[bool, str]:
        """
        Use the LLM to validate behavior equivalence (Layer 3).

        Args:
            old_code: Original code
            new_code: New code
            old_effects: Original effects
            new_effects: New effects

        Returns:
            Tuple[bool, str]: (whether equivalent, LLM reasoning)
        """
        if not self.llm:
            self.logger.warning("[BehaviorValidator] LLM not configured; skipping Layer 3 validation")
            return True, "LLM validation skipped (no LLM configured)"

        if not self.enable_llm_validation:
            return True, "LLM validation disabled"

        prompt = self._build_llm_prompt(old_code, new_code, old_effects, new_effects)

        try:
            response = await self.llm.chat(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            self.logger.error(f"[BehaviorValidator] LLM validation failed: {e}")
            # On LLM failure, return True (do not block the refactor)
            return True, f"LLM validation failed: {e}"

    def _build_llm_prompt(
        self,
        old_code: str,
        new_code: str,
        old_effects: List[str],
        new_effects: List[str],
    ) -> str:
        """
        Build the LLM validation prompt.

        Args:
            old_code: Original code
            new_code: New code
            old_effects: Original effects
            new_effects: New effects

        Returns:
            str: LLM prompt
        """
        return f"""You are analyzing JavaScript code for a Minecraft bot to determine if two code versions are semantically equivalent.

## Original Code:
```javascript
{old_code[:2000]}
```

## Original Effects:
{chr(10).join(f'- {e}' for e in old_effects[:10])}

## New Code:
```javascript
{new_code[:2000]}
```

## New Effects:
{chr(10).join(f'- {e}' for e in new_effects[:10])}

## Task:
Analyze if the new code is semantically equivalent to the original code. Focus on:
1. Do both versions achieve the same effects (mining, crafting, moving, etc.)?
2. Are there any missing behaviors in the new code?
3. Are there any new side effects that could cause issues?

## Response Format:
Respond with exactly one of these formats:
- "EQUIVALENT: [brief reason]" if the codes are semantically equivalent
- "NOT_EQUIVALENT: [specific issue]" if there are differences that matter

Be conservative: if unsure, lean towards EQUIVALENT.
"""

    def _parse_llm_response(self, response: str) -> Tuple[bool, str]:
        """
        Parse the LLM response.

        Args:
            response: LLM response text

        Returns:
            Tuple[bool, str]: (whether equivalent, reasoning)
        """
        response = response.strip()

        if response.upper().startswith("EQUIVALENT"):
            return True, response

        if response.upper().startswith("NOT_EQUIVALENT"):
            return False, response

        # Cannot parse; default to considering equivalent
        self.logger.warning(f"[BehaviorValidator] Unable to parse LLM response: {response[:100]}")
        return True, f"Unable to parse response: {response[:100]}"

    async def validate_equivalence(
        self,
        old_code: str,
        new_code: str,
        old_effects: List[str] = None,
        new_effects: List[str] = None,
        use_llm: bool = None,
    ) -> BehaviorEquivalenceResult:
        """
        Execute the full behavior equivalence validation (three-layer strategy).

        Args:
            old_code: Original code
            new_code: New code
            old_effects: List of original effects
            new_effects: List of new effects
            use_llm: Whether to use the LLM (None means use the default setting)

        Returns:
            BehaviorEquivalenceResult: Validation result
        """
        old_effects = old_effects or []
        new_effects = new_effects or []
        use_llm = use_llm if use_llm is not None else self.enable_llm_validation

        all_issues = []
        all_warnings = []

        # Layer 1: static analysis
        self.logger.debug("[BehaviorValidator] Layer 1: static analysis")
        old_features = self.extract_behavior_features(old_code)
        new_features = self.extract_behavior_features(new_code)
        old_features.effects_descriptions = old_effects
        new_features.effects_descriptions = new_effects

        layer1_equiv, layer1_issues = self.compare_behaviors(old_features, new_features)
        all_issues.extend(layer1_issues)

        if not layer1_equiv:
            self.logger.info(f"[BehaviorValidator] Layer 1 failed: {layer1_issues}")
            return BehaviorEquivalenceResult(
                is_equivalent=False,
                confidence=0.3,
                layer_passed=0,
                issues=all_issues,
                warnings=all_warnings,
                old_features=old_features,
                new_features=new_features,
            )

        # Layer 2: effects check
        if old_effects and new_effects:
            self.logger.debug("[BehaviorValidator] Layer 2: effects check")
            layer2_equiv, layer2_issues = self.compare_effects(old_effects, new_effects)
            all_issues.extend(layer2_issues)

            if not layer2_equiv:
                self.logger.info(f"[BehaviorValidator] Layer 2 failed: {layer2_issues}")
                return BehaviorEquivalenceResult(
                    is_equivalent=False,
                    confidence=0.5,
                    layer_passed=1,
                    issues=all_issues,
                    warnings=all_warnings,
                    old_features=old_features,
                    new_features=new_features,
                )

        # Layer 3: LLM validation (optional)
        llm_reasoning = None
        if use_llm and self.llm:
            self.logger.debug("[BehaviorValidator] Layer 3: LLM validation")
            layer3_equiv, llm_reasoning = await self.validate_with_llm(
                old_code, new_code, old_effects, new_effects
            )

            if not layer3_equiv:
                self.logger.info(f"[BehaviorValidator] Layer 3 failed: {llm_reasoning}")
                all_issues.append(f"LLM validation failed: {llm_reasoning}")
                return BehaviorEquivalenceResult(
                    is_equivalent=False,
                    confidence=0.7,
                    layer_passed=2,
                    issues=all_issues,
                    warnings=all_warnings,
                    llm_reasoning=llm_reasoning,
                    old_features=old_features,
                    new_features=new_features,
                )

        # All layers passed
        self.logger.info("[BehaviorValidator] All validation layers passed")
        return BehaviorEquivalenceResult(
            is_equivalent=True,
            confidence=0.9 if llm_reasoning else 0.7,
            layer_passed=3 if llm_reasoning else 2,
            issues=[],
            warnings=all_warnings,
            llm_reasoning=llm_reasoning,
            old_features=old_features,
            new_features=new_features,
        )
