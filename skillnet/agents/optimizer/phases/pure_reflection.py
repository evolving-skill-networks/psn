"""
Pure Reflection - pure psn_reflection implementation

Designed from first principles:
    psn_reflection(skill, feedback) → (δ_skill, {φ_(skill→child)})

Core ideas:
1. psn_reflection is a pure function (given the same input, always produces the same output)
2. Does not validate, filter, or skip-check (that's the caller's responsibility)
3. Only does "differentiation": analyze problems -> generate gradients

Mathematical form:
- For a failed skill A and feedback Y
- psn_reflection(A, Y) = (δ_A, {φ_(A→B) | B ∈ FaultyChildren(A)})
- where δ_A is the modification suggestion for A
- φ_(A→B) is the feedback propagated to child skill B (the local gradient of the chain rule)
"""

import logging
from dataclasses import dataclass, field

from skillnet.utils.stats_tracker import record_llm_usage

logger = logging.getLogger(__name__)
from typing import Dict, List, Set, Any, Optional, Tuple, Protocol, TYPE_CHECKING
from enum import Enum
from datetime import datetime

if TYPE_CHECKING:
    from ..knowledge.physical_constraints import ConstraintViolation

# API behavior knowledge and reasoning examples use lazy imports to avoid circular deps
# See the try block in _build_analysis_prompt


# ============================================================
# Part 1: Core data structures
# ============================================================

class GradientType(Enum):
    """Gradient types — the dimensions that need to be modified"""

    # Logic layer
    LOGIC = "logic"                      # Logic error
    CONTROL_FLOW = "control_flow"        # Control flow issue
    ERROR_HANDLING = "error_handling"    # Error handling

    # Interface layer
    PARAMETER_SEMANTIC = "parameter_semantic"  # Parameter semantic mismatch (most common)
    PARAMETER_TYPE = "parameter_type"          # Parameter type error
    RETURN_VALUE = "return_value"              # Return value issue

    # Effect layer
    PRECONDITION = "precondition"        # Precondition
    EFFECT = "effect"                    # Expected effect not achieved
    SIDE_EFFECT = "side_effect"          # Side effect issue

    # Environment layer
    ENVIRONMENT_ADAPTATION = "environment_adaptation"  # Environment adaptability
    RESOURCE_MANAGEMENT = "resource_management"        # Resource management
    PHYSICAL_CONSTRAINT = "physical_constraint"        # Physical constraint not satisfied (depth, position, distance, etc.)


@dataclass(frozen=True)
class Gradient:
    """
    Gradient — modification direction for a skill

    This is the atomic unit of psn_reflection's output.
    A skill may have multiple gradients, representing modifications needed along multiple dimensions.

    Mathematical meaning: the component of ∂L/∂skill along one dimension
    """
    gradient_type: GradientType
    magnitude: float           # Gradient magnitude [0, 1], representing the urgency of modification
    direction: str             # Textual description of the modification direction
    evidence: str              # Evidence supporting this gradient

    # Optional concrete fix suggestion
    suggested_fix: Optional[str] = None
    affected_lines: Optional[List[int]] = None

    def __repr__(self):
        return f"∇({self.gradient_type.value}, mag={self.magnitude:.2f})"


@dataclass
class SkillDelta:
    """
    Skill Delta (δ) - the complete modification suggestion for a single skill

    This is the aggregation of multiple gradients, representing the overall modification direction for a skill.

    Mathematical meaning: δ_skill = Σ gradients
    """
    skill_name: str
    gradients: List[Gradient]

    # Aggregated information
    total_magnitude: float = 0.0     # Total gradient magnitude
    primary_type: Optional[GradientType] = None  # Primary issue type

    # Metadata
    source_feedback: Optional[str] = None  # Feedback that produced this delta
    analysis_depth: int = 0                # Analysis depth (recursion level)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if self.gradients:
            self.total_magnitude = sum(g.magnitude for g in self.gradients)
            # Use the type of the largest gradient as the primary type
            max_gradient = max(self.gradients, key=lambda g: g.magnitude)
            self.primary_type = max_gradient.gradient_type

    @property
    def is_significant(self) -> bool:
        """Whether the gradient is significant (needs optimization)"""
        return self.total_magnitude > 0.3

    def __repr__(self):
        return f"δ({self.skill_name}, |∇|={self.total_magnitude:.2f}, type={self.primary_type})"


@dataclass
class PropagatedFeedback:
    """
    Propagated Feedback (φ) - feedback passed from a parent skill to a child skill

    This is the core of the chain rule:
    φ_(A→B) = the part of A's issues that B is responsible for

    Mathematical meaning: the ∂A/∂B part of ∂A/∂B · ∂L/∂A
    """
    source_skill: str          # Source skill (parent)
    target_skill: str          # Target skill (child)

    # Propagated content
    issue_description: str     # Issue description
    responsibility: str        # Responsibility the child skill should take
    expected_behavior: str     # Expected behavior
    actual_behavior: str       # Actual behavior

    # Weight and confidence
    weight: float = 1.0        # Propagation weight (proportion of responsibility for the child skill)
    confidence: float = 0.5    # Confidence

    # Context
    task: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_gradient_context(self) -> str:
        """Convert to context usable for the next layer of analysis"""
        return f"""
[Propagated from {self.source_skill}]
Issue: {self.issue_description}
Your responsibility: {self.responsibility}
Expected: {self.expected_behavior}
Actual: {self.actual_behavior}
"""


@dataclass
class ReflectionInput:
    """
    Input for psn_reflection

    Wraps all information required to perform "symbolic differentiation".
    """
    skill_name: str
    skill_code: str
    skill_description: str

    # Feedback information
    feedback_content: str
    feedback_type: str  # error, critique, semantic_warning, etc.

    # Execution data
    execution_traces: List[Dict[str, Any]]
    pre_state: Optional[Dict[str, Any]] = None
    post_state: Optional[Dict[str, Any]] = None

    # Child skill information
    children: List[str] = field(default_factory=list)
    children_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Propagated feedback from the parent skill (if any)
    propagated_feedback: Optional[PropagatedFeedback] = None

    # in-game chat log during execution
    # Chat messages often contain diagnostic information (e.g., "I need at least a stone_pickaxe...")
    # that is crucial for correct problem diagnosis
    chat_log: str = ""

    # Task-specific wrapper flag: directs Phase 1 to focus on child_issues
    is_task_specific: bool = False

    # Refactor-derived wrapper flags. When set, Phase 1 is instructed to
    # consider redirecting fixes to `covered_by` rather than expanding the
    # wrapper into an inline implementation. Genuine wrapper bugs (wrong arg
    # mapping, validation) remain in-scope for self_issues.
    is_covered: bool = False
    covered_by: Optional[str] = None


@dataclass
class ReasoningChain:
    """
    Explicit reasoning chain — records the LLM's reasoning process

    Used to improve debuggability and interpretability:
    - Track which knowledge was retrieved
    - Record reasoning steps
    - Explain how knowledge was applied to reach conclusions
    """
    # Retrieved knowledge items
    retrieved_knowledge: List[Dict[str, str]] = field(default_factory=list)

    # Structure of the LLM's reasoning
    error_understanding: str = ""       # Understanding of the error
    knowledge_applied: List[str] = field(default_factory=list)  # List of applied knowledge
    inference: str = ""                 # Inference process

    # Raw text (backward compatibility)
    raw_text: str = ""


@dataclass
class ReflectionOutput:
    """
    Output of psn_reflection

    This is the concrete realization of (δ, {φ}):
    - delta: modification suggestion for the current skill
    - propagated_feedbacks: feedback to propagate to child skills
    - faulty_children: the set of child skills identified as problematic
    """
    skill_name: str

    # δ - modification suggestion for the current skill
    delta: SkillDelta

    # {φ} - feedback propagated to child skills
    propagated_feedbacks: Dict[str, PropagatedFeedback]

    # FaultyChildren - child skills that need further analysis
    faulty_children: Set[str]

    # Analysis metadata
    analysis_successful: bool = True
    confidence: float = 0.5
    reasoning: str = ""

    # Explicit reasoning chain (optional, for debugging and interpretability)
    reasoning_chain: Optional[ReasoningChain] = None

    # Constraint violations list (for optimizer_impl to convert to SkillFeedback)
    constraint_violations: List['ConstraintViolation'] = field(default_factory=list)

    # children identified in calling_pattern_issues (no recursive analysis needed)
    caller_fix_children: Set[str] = field(default_factory=set)

    # Parameter corrections from Phase 1 RCA (task-scoped feedback for PureLLM)
    parameter_corrections: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def needs_propagation(self) -> bool:
        """Whether downward propagation is required"""
        return len(self.faulty_children) > 0

    @property
    def self_fix_only(self) -> bool:
        """Whether only self-fix is required"""
        return self.delta.is_significant and not self.needs_propagation




# ============================================================
# Part 3: PureReflection interface
# ============================================================

class ReflectionAnalyzer(Protocol):
    """
    Analyzer protocol — defines how to perform "differentiation"

    Different analyzers can have different implementations:
    - PatternBasedAnalyzer: fast pattern-matching analysis
    - LLMAnalyzer: deep LLM-based analysis
    - HybridAnalyzer: hybrid analysis
    """

    def analyze(
        self,
        input: ReflectionInput,
    ) -> ReflectionOutput:
        """Run the analysis and return (δ, {φ})"""
        ...


class PureReflection:
    """
    Pure psn_reflection implementation

    Core principles:
    1. This is a pure function — does not mutate any state
    2. Single responsibility — only performs "differentiation"
    3. Composable — multiple analyzers can be combined

    Usage:
        reflection = PureReflection(analyzer=my_analyzer)

        # Single reflection
        output = reflection.reflect(input)

        # Output contains:
        # - delta: modification suggestion for the current skill
        # - propagated_feedbacks: feedback for child skills
        # - faulty_children: child skills that need further analysis
    """

    def __init__(self, analyzer: ReflectionAnalyzer):
        """
        Initialize PureReflection

        Args:
            analyzer: the analyzer that actually performs the analysis
        """
        self.analyzer = analyzer

    def reflect(
        self,
        input: ReflectionInput,
    ) -> ReflectionOutput:
        """
        Run psn_reflection

        This is the core "symbolic differentiation" operation:
            psn_reflection(skill, feedback) → (δ, {φ})

        Args:
            input: reflection input (contains skill and feedback info)

        Returns:
            ReflectionOutput: (δ, {φ}) - delta and propagated feedbacks
        """
        # Delegate to the analyzer to perform the actual analysis
        output = self.analyzer.analyze(input)

        return output



# ============================================================
# Part 4: Concrete analyzer implementations
# ============================================================


class LLMAnalyzer:
    """
    LLM-based deep analyzer

    Uses an LLM for complex root cause analysis, able to:
    1. Understand code semantics
    2. Analyze execution traces
    3. Identify complex parameter semantic mismatches
    4. Account for environmental factors
    """

    def __init__(self, llm, logger=None, pure_reasoning=False,
                 include_reasoning_examples=False,
                 use_factual_primitive_doc=False):
        self.llm = llm
        self.logger = logger
        self._pure_reasoning = pure_reasoning
        # Gates the reasoning_examples_section build. Default False — an
        # ablation showed no diagnostic-correctness change with the guides
        # enabled.
        self._include_reasoning_examples = include_reasoning_examples
        # Gate factual primitive-doc injection. Default False; when True,
        # doc strings for primitives detected in skill_code are appended to
        # the analysis prompt (see
        # skillnet/agents/optimizer/phases/factual_primitive_doc).
        self._use_factual_primitive_doc = use_factual_primitive_doc
        # Domain knowledge (set externally for from_domain path)
        self._domain_knowledge = None
        # Knowledge retrieval diagnostic logger (set externally by engine)
        self._kr_logger = None

    def analyze(
        self,
        input: ReflectionInput,
    ) -> ReflectionOutput:
        """Run deep analysis using the LLM"""
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(input)

        # Invoke the LLM
        try:
            response = self._invoke_llm(prompt)

            result = self._parse_response(input, response)

            # Retry once with a repair prompt if JSON parsing failed
            if not result.analysis_successful and response:
                repair_prompt = (
                    "Your previous response was not valid JSON. "
                    "Return ONLY a JSON object with no extra text, no markdown, no explanation.\n\n"
                    f"Original request:\n{prompt}\n\n"
                    f"Your previous response (INVALID):\n{response[:2000]}"
                )
                retry_response = self._invoke_without_thinking(repair_prompt)
                retry_result = self._parse_response(input, retry_response)
                if retry_result.analysis_successful:
                    if self.logger:
                        self.logger.info("[Phase 1 Analyzer] JSON repair retry succeeded")
                    return retry_result

            return result
        except Exception as e:
            # Return empty result on failure
            return ReflectionOutput(
                skill_name=input.skill_name,
                delta=SkillDelta(skill_name=input.skill_name, gradients=[]),
                propagated_feedbacks={},
                faulty_children=set(),
                analysis_successful=False,
                reasoning=f"LLM analysis failed: {e}",
            )

    def _build_analysis_prompt(
        self,
        input: ReflectionInput,
    ) -> str:
        """Build the LLM analysis prompt"""
        # Initialize knowledge tracking (used for the explicit reasoning chain)
        self._retrieved_knowledge: List[Dict[str, str]] = []

        # Knowledge retrieval diagnostic logger (may be None)
        kr = self._kr_logger

        if kr:
            kr.info("=" * 60)
            kr.info(f"[Analysis] skill={input.skill_name} feedback_type={input.feedback_type}")
            kr.info(f"[Analysis] feedback={input.feedback_content[:200]}...")
            if input.chat_log:
                kr.info(f"[Analysis] chat_log={input.chat_log[:200]}...")

        env_section = ""
        primitive_section = ""
        biome_section = ""  # P0: biome environment info

        if self._domain_knowledge:
            # Domain-aware path: all environment knowledge from DomainKnowledge

            # Environment context (biome, etc.)
            if input.pre_state and isinstance(input.pre_state, dict):
                biome_section = self._domain_knowledge.get_environment_context(
                    input.pre_state
                )

            # Environment rules
            rules = self._domain_knowledge.get_environment_rules(
                input.feedback_content
            )
            if rules:
                rules_text = "\n".join(f"- {r}" for r in rules)
                env_section = f"\n\n**Relevant Environment Rules:**\n{rules_text}"

            # Primitive knowledge (error-specific)
            prim = self._domain_knowledge.get_primitive_knowledge(
                input.feedback_content, input.skill_code or ""
            )
            if prim:
                primitive_section = f"\n\n{prim}"

            # Full primitive context reference. Gives Phase 1 the same primitive
            # knowledge as Phase 2, so it doesn't recommend changes that conflict
            # with primitive behavior.
            try:
                dk = self._domain_knowledge
                ctx = dk.load_control_primitive_context(for_optimizer=True) if dk else []
                if ctx:
                    primitive_section += (
                        "\n\n**Control Primitives Reference (full implementations):**\n"
                        "```javascript\n"
                        + "\n\n".join(ctx)
                        + "\n```"
                    )
            except Exception:
                pass  # Non-critical: Phase 1 still works without full context
        else:
            logger.warning(
                "[PureReflection] No domain_knowledge set — environment "
                "knowledge will be empty. Use PSNAgent.from_domain() to wire."
            )

        # Log environment knowledge sections
        if kr:
            if biome_section:
                kr.info(f"[Biome] {biome_section[:300]}")
            if env_section:
                kr.info(f"[EnvRules] {env_section[:500]}")
            if primitive_section:
                kr.info(f"[Primitive] {primitive_section[:500]}")

        children_section = ""
        if input.children:
            children_lines = []
            for c in input.children:
                info = input.children_info.get(c, {})
                desc = info.get("description", "")
                params = info.get("parameters", {})
                sr = info.get("success_rate", 0.0)

                # Build parameter signature
                param_parts = []
                for pname, pmeta in params.items():
                    ptype = pmeta.get("type", "any")
                    param_parts.append(f"{pname}: {ptype}")
                param_sig = f"({', '.join(param_parts)})" if param_parts else "()"

                line = f"- {c}{param_sig}: {desc} [success_rate: {sr:.0%}]"

                # Show last execution failure info
                last_exec = info.get("last_execution", {})
                if last_exec and not last_exec.get("success", True):
                    err = last_exec.get("error_message", "")[:150]
                    line += f"\n  Last execution FAILED: {err}"

                children_lines.append(line)
            children_text = "\n".join(children_lines)
            children_section = f"\n\n**Child Skills (with parameter signatures):**\n{children_text}"

        # Add propagated feedback from the parent skill (key information from Top-Down propagation)
        propagated_section = ""
        if input.propagated_feedback:
            pf = input.propagated_feedback
            propagated_section = f"""

**Feedback from Parent Skill ({pf.source_skill}):**
This skill was identified as problematic by its parent. Pay special attention to this feedback.
- Issue: {pf.issue_description}
- Responsibility: {pf.responsibility}
- Weight: {pf.weight:.2f}
- Context: {pf.to_gradient_context()[:500] if hasattr(pf, 'to_gradient_context') else 'N/A'}"""

        # New: build the execution-state section (including call args and inventory changes)
        execution_state_section = self._build_execution_state_section(input)

        # add the chat_log section — contains runtime diagnostic info
        # For example "I need at least a stone_pickaxe to mine deepslate_iron_ore! Skip it!"
        chat_log_section = ""
        if input.chat_log:
            chat_log_section = f"""

**Execution Chat Log:**
This is the bot's chat output during execution. Pay special attention to diagnostic messages like "I need at least..." or "Skip it!".
```
{input.chat_log[:2000]}
```"""

        # Phase 2: add API behavior knowledge and reasoning examples (lazy import to avoid circular deps)
        api_knowledge_section = ""
        reasoning_examples_section = ""
        try:
            if self._domain_knowledge:
                # Domain-aware path: use domain knowledge for error retrieval
                search_text = input.feedback_content
                if input.chat_log:
                    search_text = f"{input.feedback_content}\n{input.chat_log}"

                # Try detailed retrieval first (fast-path detection + indexed search)
                detailed = self._domain_knowledge.get_detailed_knowledge_for_error(
                    input.feedback_content, input.skill_code or "",
                    input.chat_log or "",
                    kr_logger=kr,
                )
                if detailed:
                    api_knowledge_section = f"\n\n{detailed}"
                    self._retrieved_knowledge.append({
                        "source": "domain_detailed_knowledge",
                        "content": detailed[:500] + "..."
                                   if len(detailed) > 500 else detailed
                    })
                    if kr:
                        kr.info(f"[DomainDetailed] chars={len(detailed)}")
                        kr.info(f"[DomainDetailed] {detailed[:500]}...")
                else:
                    # Fall back to basic retrieval
                    error_knowledge = self._domain_knowledge.get_knowledge_for_error(
                        search_text, input.skill_code or ""
                    )
                    if error_knowledge:
                        api_knowledge_section = "\n\n" + "\n\n".join(
                            str(k) for k in error_knowledge
                        )
                        self._retrieved_knowledge.append({
                            "source": "domain_knowledge",
                            "content": api_knowledge_section[:500] + "..."
                                       if len(api_knowledge_section) > 500
                                       else api_knowledge_section
                        })
                        if kr:
                            kr.info(f"[DomainBasic] items={len(error_knowledge)} "
                                    f"chars={len(api_knowledge_section)}")
                    elif kr:
                        kr.info("[DomainKnowledge] no results")

                # Domain reasoning examples — gated by
                # OptimizationConfig.include_reasoning_examples (default False).
                # Catalog retained in source for optionality and future research.
                if self._include_reasoning_examples:
                    examples = self._domain_knowledge.get_reasoning_examples(
                        search_text, max_examples=2
                    )
                    if examples:
                        reasoning_examples_section = f"\n\n{examples}"
                        self._retrieved_knowledge.append({
                            "source": "domain_reasoning_examples",
                            "content": examples[:500] + "..."
                                       if len(examples) > 500 else examples
                        })
                        if kr:
                            kr.info(f"[DomainReasoning] chars={len(examples)}")
                elif kr:
                    kr.info("[DomainReasoning] disabled by include_reasoning_examples=False")
            # No else — if domain knowledge is not wired, knowledge sections stay empty.
        except KeyboardInterrupt:
            # Don't swallow keyboard interrupts silently, but don't crash either
            logging.getLogger(__name__).warning(
                "[LLMAnalyzer] Knowledge retrieval interrupted, continuing without knowledge"
            )
        except Exception as e:
            # If knowledge retrieval fails, do not block the main flow
            pass

        # Composable skills section for Phase 1.
        # Without this, Phase 1 gradients always point to primitives (placeItem)
        # instead of learned skills (placeCraftingTable). LLM retrieval selects
        # top-5 relevant skills from the full graph based on the error context.
        composable_section = ""
        try:
            sgm = getattr(self, '_skill_graph_manager', None)
            if sgm:
                all_names = sgm.get_all_skill_names(include_task_specific=False)
                # Exclude the skill being analyzed
                candidates = [n for n in all_names if n != input.skill_name]
                # Only include skills with at least 1 execution
                qualified = []
                for name in candidates:
                    node = sgm.get_node(name)
                    if node and node.statistics and node.statistics.total_executions >= 1:
                        desc = (node.description or "")[:60]
                        qualified.append((name, desc))

                if qualified:
                    if len(qualified) <= 5:
                        selected_skills = qualified
                    else:
                        # LLM retrieval: pick top 5 most relevant
                        selected_skills = self._select_composable_for_phase1(
                            input.feedback_content, input.skill_name, qualified
                        )

                    if selected_skills:
                        lines = ["**Available learned skills (consider calling instead of reimplementing):**"]
                        for name, desc in selected_skills:
                            lines.append(f"- {name}: {desc}")
                        composable_section = "\n".join(lines)

                        if kr:
                            names = [n for n, _ in selected_skills]
                            kr.info(f"[Composable Phase1] {input.skill_name}: "
                                    f"{len(selected_skills)} skills: {names}")
                    elif kr:
                        kr.info(f"[Composable Phase1] {input.skill_name}: "
                                f"no skills selected (qualified={len(qualified)})")
                elif kr:
                    kr.info(f"[Composable Phase1] {input.skill_name}: "
                            f"no qualified skills (total={len(all_names)})")
        except Exception as e:
            logger.debug(f"[Composable Phase1] Failed: {e}")

        # Log knowledge retrieval summary
        if kr:
            sections = {
                "biome": len(biome_section),
                "env_rules": len(env_section),
                "primitive": len(primitive_section),
                "api_knowledge": len(api_knowledge_section),
                "reasoning": len(reasoning_examples_section),
                "composable": len(composable_section),
            }
            active = {k: v for k, v in sections.items() if v > 0}
            kr.info(f"[Summary] active_sections={active} "
                    f"total_chars={sum(sections.values())}")
            # Log pre_state availability for inventory diagnosis
            has_pre = bool(input.pre_state)
            pre_inv = (input.pre_state or {}).get("inventory", {}) if has_pre else {}
            _pre_inv_for_log = (input.pre_state or {}).get("inventory", {})
            kr.info(f"[Inventory] pre_state={'present' if input.pre_state else 'NONE'} "
                    f"pre_state_type={type(input.pre_state).__name__} "
                    f"pre_state_truthy={bool(input.pre_state)} "
                    f"inv_items={len(_pre_inv_for_log)} "
                    f"sample={list(_pre_inv_for_log.items())[:3] if _pre_inv_for_log else '[]'}")

        # Wrapper guidance for task-specific skills
        wrapper_guidance = ""
        if input.is_task_specific:
            wrapper_guidance = """
**WRAPPER SKILL GUIDANCE:**
This skill is a task-specific wrapper generated by the planner. It only orchestrates calls to child skills.
- DO NOT suggest fixes to this wrapper's own logic (self_issues) unless it passes wrong arguments
- FOCUS on identifying which child skill(s) failed and why → put analysis in child_issues
- The optimization of actual logic should happen in the child skills, not here
- Valid self_issues for wrappers: wrong call arguments, wrong call order, missing error propagation
"""
        # Refactor-derived wrapper guidance.
        # When a skill is covered by a sibling/parametric refactor (is_covered=True),
        # it's a 3-8 line wrapper around its covered_by parent skill. The wrapper's
        # only responsibility is transforming inputs / shaping outputs around the
        # covered_by call. If failures are in the covered_by's logic, the fix
        # should redirect there — NOT expand this wrapper into an inline
        # implementation, which would destroy the refactor architecture.
        if input.is_covered and input.covered_by:
            wrapper_guidance += f"""
**REFACTOR-DERIVED WRAPPER GUIDANCE:**
This skill is a thin wrapper around `{input.covered_by}` (created by a sibling
or parametric refactor). Its job is parameter transformation, input validation,
or output shaping AROUND the call to `{input.covered_by}`.
- If the root cause of the failure is inside `{input.covered_by}`'s logic, your
  gradient/correction should target `{input.covered_by}` (use child_issues for
  that skill) — NOT expand this wrapper into an inline reimplementation.
- Valid self_issues for this wrapper: wrong argument mapping to `{input.covered_by}`,
  missing pre/post-validation, incorrect default values.
- Expanding to >15 lines almost certainly destroys the refactor; consider whether
  the inner skill needs the fix instead.
"""

        # Determine language and domain name for prompt
        if self._domain_knowledge:
            skill_lang = self._domain_knowledge.get_skill_language()
            domain_name = self._domain_knowledge.get_domain_name()
        else:
            skill_lang = ""
            domain_name = "domain"

        # Factual primitive-doc section (gated).
        factual_primitive_section = ""
        if self._use_factual_primitive_doc:
            try:
                from .factual_primitive_doc import build_factual_primitive_section
                factual_primitive_section = build_factual_primitive_section(
                    input.skill_code or ""
                )
                if kr and factual_primitive_section:
                    kr.info(
                        f"[FactualPrimitiveDoc] chars={len(factual_primitive_section)}"
                    )
            except Exception as e:
                logger.debug(f"[FactualPrimitiveDoc] build failed: {e}")

        return f"""Analyze why this skill failed and determine the modification direction.
{wrapper_guidance}
**Skill:** {input.skill_name}

**Code:**
```{skill_lang}
{input.skill_code[:8000]}
```

**Feedback:**
{input.feedback_content}

**Feedback Type:** {input.feedback_type}
{execution_state_section}
{chat_log_section}
{biome_section}
{children_section}
{env_section}
{primitive_section}
{factual_primitive_section}
{propagated_section}
{api_knowledge_section}
{composable_section}
{reasoning_examples_section}

**Analysis Tasks:**
1. Identify the root cause of the failure
2. Determine the attribution — THREE possibilities:
   (a) THIS skill's own logic has a bug (→ self_issues)
   (b) THIS skill is CALLING a child incorrectly — wrong arguments, wrong types, wrong data, missing precondition setup (→ calling_pattern_issues). Check child parameter signatures above.
   (c) A child skill's OWN implementation has a bug (→ child_issues)
   NOTE: (b) and (c) can BOTH apply if the calling code is wrong AND the child also has a separate internal bug.
3. ENUMERATE ALL DISTINCT ISSUES you can identify in this skill — not just the most-visible one.
   A skill may have multiple independent bugs stacked; under-enumerating here means
   hidden bugs persist across optimization cycles and exhaust the retry budget.
4. For each identified issue, specify:
   - The type of gradient — choose the MOST SPECIFIC applicable from this taxonomy:
     * `logic`                  — flawed control logic / wrong branching / bad algorithm
     * `control_flow`           — infinite loop, missing break, unreachable code
     * `error_handling`         — missing try/catch, swallowed errors, bad recovery
     * `parameter_semantic`     — argument VALUE wrong for task (wrong count, wrong variant, out-of-range)
     * `parameter_type`         — argument TYPE mismatch vs API signature
     * `return_value`           — not checking return value, misusing it, wrong expected type
     * `precondition`           — required setup missing
     * `effect`                 — desired state change did not occur
     * `side_effect`            — unintended state change
     * `environment_adaptation` — code doesn't adapt to current biome/Y/time/inventory
     * `resource_management`    — running out of / wasting resources
     * `physical_constraint`    — spatial/distance/depth constraint violated
   - The magnitude (0.0 to 1.0, higher = more urgent)
   - The direction (what needs to change)
   - The suggested_fix (REQUIRED: concrete code modification suggestions)
5. Evaluate the CALL ARGUMENTS passed to this skill — the planner may have inferred
   incorrect parameter values. Compare call_args against parameter defaults and the
   execution outcome. If a non-default parameter value contributed to the failure,
   report it in parameter_corrections. Only include corrections where you are
   confident the parameter value caused or contributed to the failure.
6. Document your reasoning chain (which knowledge you used and how)

**CRITICAL - Reasoning from Facts:**
You have access to {domain_name} facts, API behaviors, and constraints in the knowledge sections above.
Your job is to REASON from these facts, NOT to copy code patterns or memorized solutions.

For each problem:
1. Identify what facts/constraints from the knowledge apply to this situation
2. Reason step-by-step about why the error occurred based on those facts
3. Determine what action satisfies all the constraints
4. Write ORIGINAL code based on your reasoning (not pattern-matched from examples)

Do NOT simply copy suggested fixes from reasoning guides - derive your solution from the underlying facts.

**suggested_fix QUALITY GUIDE (your suggested_fix drives the code generator — be precise):**

BAD (too vague — code generator will guess wrong):
  "Add crafting table placement logic before crafting"
  "Handle the case where the block is not found"

GOOD (diagnosis + direction, NOT code):
  "Root cause: no placed crafting_table block within range (knowledge: craftItem_no_table).
   Direction: verify table exists before crafting; if absent, place one at safe offset position.
   Reference: [craftItem_no_table, placeItem_internal_capabilities]"

GOOD (using retrieved knowledge):
  "Root cause: blockUpdate timeout — target position occupied by bot entity
   (knowledge: placeItem_bot_at_target, implication: offset ±1 guaranteed safe).
   Direction: place at offset from bot position to ensure target is unoccupied."

Rules:
- Reference SPECIFIC knowledge items you used to reach your conclusion
- Describe the PROBLEM and DIRECTION clearly, not specific function calls or code
- If the fix involves multiple steps, number them
- If a control primitive is relevant, name it but let the code generator handle the call details

**Callable names must exist.** Every function name you put in `suggested_fix`,
`direction`, or any other freeform field MUST appear verbatim in one of the
explicit lists already shown above:
  - The Control Primitives Reference (full implementations shown earlier),
  - The Available learned skills composable section,
  - Functions defined inside the skill's own code (shown above).
If no existing callable matches the behavior you want, describe the desired
behavior in prose rather than naming a function that doesn't exist (the
downstream code generator will reject any patch that calls a function not
present in those lists, and your suggestion will be wasted).

Return JSON:
{{
    "reasoning": {{
        "error_understanding": "What exactly went wrong (1-2 sentences)",
        "knowledge_applied": ["List the specific knowledge items you used from the retrieved knowledge above"],
        "inference": "How you derived the fix from the knowledge (explain your reasoning)"
    }},
    "self_issues": [
        {{
            "gradient_type": "one of: logic|control_flow|error_handling|parameter_semantic|parameter_type|return_value|precondition|effect|side_effect|environment_adaptation|resource_management|physical_constraint",
            "magnitude": 0.0-1.0,
            "direction": "what needs to change",
            "evidence": "supporting evidence from feedback",
            "suggested_fix": "REQUIRED: specific code changes to make"
        }}
    ],
    "calling_pattern_issues": [
        {{
            "child_skill": "name of the child skill being called incorrectly by THIS skill",
            "issue_description": "How THIS skill calls the child incorrectly (wrong args, wrong types, wrong semantics)",
            "evidence": "Code line or trace showing the argument mismatch",
            "suggested_fix": "How to fix the calling code in THIS skill (NOT the child)"
        }}
    ],
    "child_issues": [
        {{
            "child_skill": "name",
            "issue_description": "...",
            "responsibility": "...",
            "expected_behavior": "What the child skill should have achieved",
            "actual_behavior": "What actually happened or the error observed",
            "weight": 0.0-1.0
        }}
    ],
    "parameter_corrections": [
        {{
            "target_skill": "REQUIRED: the skill whose parameter value was wrong. When THIS skill calls a child with wrong arguments (the common wrapper case), target_skill is the CHILD's name (e.g. 'exploreUntilSkill'). When the bug is in THIS skill's own signature/usage, target_skill is THIS skill's own name. Without this field the correction cannot be matched to the right skill at planner time.",
            "param_name": "name of the parameter that was wrong",
            "passed_value": "the value that was used",
            "suggested_value": "what the value should have been",
            "reason": "why the passed value was wrong and why the suggested value is better",
            "confidence": 0.0-1.0
        }}
    ]
}}"""

    def _select_composable_for_phase1(
        self,
        feedback: str,
        skill_name: str,
        qualified: list,  # [(name, desc), ...]
        max_n: int = 5,
    ) -> list:
        """Use a lightweight LLM call to select the most relevant composable skills.

        Phase 1 needs to know which learned skills are available so
        its gradient direction points to calling existing skills rather than
        reimplementing with primitives.

        Args:
            feedback: The error/feedback text
            skill_name: The skill being optimized
            qualified: List of (name, desc) tuples for all qualified composable skills
            max_n: Maximum skills to select

        Returns:
            List of (name, desc) tuples for the selected skills
        """
        skills_str = "\n".join(f"- {name}: {desc}" for name, desc in qualified)
        prompt = (
            f"Given this error and skill, select the {max_n} most relevant "
            f"existing skills that could help fix the problem. "
            f"Return ONLY a JSON array of {max_n} skill names.\n\n"
            f"Error: {feedback[:300]}\n"
            f"Skill being optimized: {skill_name}\n\n"
            f"Available skills:\n{skills_str}"
        )
        try:
            from langchain.schema import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            record_llm_usage(response, process_type="reflect", function_name="pure_reflection._select_composable_for_phase1", skill_name=skill_name)
            content = response.content if hasattr(response, 'content') else str(response)

            import json as _json
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                selected_names = _json.loads(match.group(0))[:max_n]
                # Map back to (name, desc) tuples
                name_to_desc = {n: d for n, d in qualified}
                return [(n, name_to_desc[n]) for n in selected_names if n in name_to_desc]
        except Exception as e:
            logger.debug(f"[Composable Phase1] LLM selection failed: {e}")

        # Fallback: return first max_n by original order
        return qualified[:max_n]

    def _build_execution_state_section(self, input: ReflectionInput) -> str:
        """
        Build the execution-state section, including call args and inventory changes

        This information helps the LLM judge accurately:
        1. Whether the call arguments are correct (detect parameter semantic issues)
        2. Whether the child skill truly completed the task (verified via inventory changes)
        """
        sections = []

        # 1. Call arguments (extracted from the most recent execution record)
        if input.execution_traces:
            latest_trace = input.execution_traces[-1] if input.execution_traces else {}
            call_args = latest_trace.get("call_args", {})
            if call_args:
                # Format arguments, handling various types
                args_parts = []
                for k, v in call_args.items():
                    if isinstance(v, str):
                        args_parts.append(f'{k}="{v}"')
                    elif isinstance(v, list):
                        args_parts.append(f'{k}={v[:3]}{"..." if len(v) > 3 else ""}')
                    else:
                        args_parts.append(f'{k}={v}')
                args_str = ", ".join(args_parts)
                sections.append(f"**Call Arguments:** {args_str}")

        # 2. Inventory (absolute + changes)
        # Use `is not None` check — pre_state may be {} (empty dict from
        # SkillExecutionTrace fallback) which is falsy but still valid.
        _pre = input.pre_state
        _post = input.post_state
        has_pre = _pre is not None and bool(_pre)
        has_post = _post is not None and bool(_post)
        if has_pre or has_post:
            pre_inv = (input.pre_state or {}).get("inventory", {})
            post_inv = (input.post_state or {}).get("inventory", {})

            # Ensure dict types
            if not isinstance(pre_inv, dict):
                pre_inv = {}
            if not isinstance(post_inv, dict):
                post_inv = {}

            # Bot position state: without this, the LLM tends to hallucinate
            # the environment from chat log fragments (e.g. claiming the bot is
            # in an ocean biome when it is actually underground in solid stone).
            # The position is present in pre_state but must be extracted into
            # the prompt for the model to use it. Including it eliminated the
            # environment-hallucination failures in A/B testing.
            pre_position = (input.pre_state or {}).get("position", {})
            if isinstance(pre_position, dict) and pre_position:
                px = pre_position.get("x")
                py = pre_position.get("y")
                pz = pre_position.get("z")
                # World-geometry config: the surface/underground heuristic only
                # makes sense for domains with a vertical (Y) axis. Pull
                # surface_y / underground_threshold from domain knowledge so
                # that a 2D domain (has_vertical=False) skips the prose entirely.
                geometry = (
                    self._domain_knowledge.get_world_geometry_config()
                    if getattr(self, "_domain_knowledge", None) is not None
                    else {"has_vertical": False}
                )
                if py is not None and geometry.get("has_vertical"):
                    try:
                        py_int = int(py)
                        # Overworld surface y is approximately 63 in most biomes.
                        # Use a conservative threshold (y < 58) for "underground"
                        # to avoid false positives in mountain biomes.
                        surface_y_approx = int(geometry["surface_y"])
                        underground_threshold = int(geometry["underground_threshold"])
                        is_underground = py_int < underground_threshold
                        depth_below_surface = max(0, surface_y_approx - py_int)
                        # Try to extract biome if available; otherwise note it's not provided.
                        biome = (input.pre_state or {}).get("biome")
                        biome_str = f", biome={biome}" if biome else ""
                        try:
                            x_str = f"{int(px)}" if px is not None else "?"
                            z_str = f"{int(pz)}" if pz is not None else "?"
                        except (ValueError, TypeError):
                            x_str, z_str = str(px), str(pz)
                        sections.append(
                            f"**Bot Position State (CRITICAL — use this to verify environmental claims):**\n"
                            f"  Position: x={x_str}, y={py_int}, z={z_str}{biome_str}\n"
                            f"  Approximate overworld surface y: ~{surface_y_approx}\n"
                            f"  is_underground: {is_underground} "
                            f"(y={py_int} {'<' if is_underground else '≥'} {underground_threshold})\n"
                            f"  Depth below surface: {depth_below_surface} blocks\n"
                            f"  Note: Trees only spawn on the surface (y≈{surface_y_approx-1}+). "
                            f"If is_underground=True, the bot CANNOT find trees from this position "
                            f"without first ascending to the surface."
                        )
                    except (ValueError, TypeError, KeyError):
                        pass  # Bad position data or missing geometry keys, skip silently

            # Show absolute inventory before execution.
            # Without this, Phase 1 can't tell if the skill should skip
            # (e.g., ensureWoodLogs mining when 50 logs already exist).
            # Test: "check inventory first" gradient 20% → 100% with this.
            if pre_inv:
                # Sort by count descending, show top 10
                sorted_items = sorted(pre_inv.items(),
                                      key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                                      reverse=True)[:10]
                inv_parts = [f"{item}: {count}" for item, count in sorted_items
                             if isinstance(count, (int, float)) and count > 0]
                if inv_parts:
                    sections.append("**Inventory Before Execution:**\n  " + ", ".join(inv_parts))

            changes = []
            all_items = set(pre_inv.keys()) | set(post_inv.keys())
            for item in sorted(all_items):
                before = pre_inv.get(item, 0)
                after = post_inv.get(item, 0)
                # Ensure numeric types
                if not isinstance(before, (int, float)):
                    before = 0
                if not isinstance(after, (int, float)):
                    after = 0
                if before != after:
                    delta = after - before
                    sign = "+" if delta > 0 else ""
                    changes.append(f"  {item}: {before} → {after} ({sign}{delta})")

            if changes:
                # Limit the number shown so the prompt doesn't get too long
                if len(changes) > 10:
                    displayed = changes[:10]
                    displayed.append(f"  ... and {len(changes) - 10} more items")
                    changes = displayed
                sections.append("**Inventory Changes During Execution:**\n" + "\n".join(changes))
            else:
                sections.append("**Inventory Changes:** None (no changes detected)")

        # 3. Summary of child skill execution results (if children_info contains execution data)
        if input.children_info:
            child_results = []
            for child_name, info in input.children_info.items():
                if isinstance(info, dict):
                    last_exec = info.get("last_execution", {})
                    if last_exec and isinstance(last_exec, dict):
                        success = last_exec.get("success")
                        if success is not None:
                            if success:
                                status = "✓ Success"
                            else:
                                err_msg = last_exec.get('error_message', 'Unknown error')
                                if err_msg and len(err_msg) > 50:
                                    err_msg = err_msg[:50] + "..."
                                status = f"✗ Failed: {err_msg}"

                            # Show inventory changes
                            inv_changes = last_exec.get("inventory_changes", {})
                            if inv_changes and isinstance(inv_changes, dict):
                                inv_parts = []
                                for k, v in list(inv_changes.items())[:5]:
                                    sign = '+' if v > 0 else ''
                                    inv_parts.append(f"{k}: {sign}{v}")
                                inv_str = ", ".join(inv_parts)
                                child_results.append(f"  - {child_name}: {status} | Inventory: {inv_str}")
                            else:
                                child_results.append(f"  - {child_name}: {status}")

            if child_results:
                sections.append("**Child Skill Execution Results:**\n" + "\n".join(child_results))

        if sections:
            return "\n\n" + "\n\n".join(sections)
        return ""

    def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM (with retry logic)"""
        if not hasattr(self.llm, 'invoke'):
            return ""

        from langchain.schema import HumanMessage
        import time

        # Retry logic (up to 3 attempts)
        for attempt in range(3):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                record_llm_usage(response, process_type="reflect", function_name="pure_reflection._invoke_llm")
                content = response.content if hasattr(response, 'content') else str(response)

                # Check for empty or too-short response
                if not content or len(content.strip()) < 10:
                    if self.logger:
                        self.logger.warning(
                            f"[Phase 1 Analyzer] Empty or too short response from LLM "
                            f"(attempt {attempt + 1}/3): {len(content) if content else 0} chars"
                        )
                    if attempt < 2:
                        time.sleep(3)  # Wait 3 seconds before retrying
                        continue
                    else:
                        raise ValueError(f"LLM returned empty response after 3 attempts")

                # Successfully obtained a valid response
                if attempt > 0 and self.logger:
                    self.logger.info(f"[Phase 1 Analyzer] LLM call succeeded on attempt {attempt + 1}/3")

                # Fix 2G: Log response metadata and detect truncation
                if hasattr(response, 'response_metadata'):
                    meta = response.response_metadata
                    token_usage = meta.get('token_usage', {})
                    if self.logger:
                        self.logger.info(
                            f"[Phase 1 Analyzer] LLM response: "
                            f"finish={meta.get('finish_reason')} "
                            f"prompt_tok={token_usage.get('prompt_tokens')} "
                            f"completion_tok={token_usage.get('completion_tokens')} "
                            f"chars={len(content)}"
                        )
                    # Detect thinking model context exhaustion
                    if meta.get('finish_reason') == 'length':
                        if self.logger:
                            self.logger.warning(
                                "[Phase 1 Analyzer] finish_reason=length — "
                                "thinking tokens likely exhausted context. "
                                "Retrying with thinking disabled."
                            )
                        return self._invoke_without_thinking(prompt)

                return content

            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"[Phase 1 Analyzer] LLM invocation failed (attempt {attempt + 1}/3): {e}"
                    )
                if attempt < 2:
                    time.sleep(3)
                else:
                    # Final attempt failed; re-raise
                    raise

        return ""

    def _invoke_without_thinking(self, prompt: str) -> str:
        """Retry LLM call with thinking disabled for structured output.

        Uses ``llm.bind()`` to create a new RunnableBinding with
        ``enable_thinking: false`` — this does not mutate the original LLM
        and is silently ignored by non-thinking models.
        """
        try:
            llm_no_think = self.llm.bind(extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            })
        except Exception:
            # Model doesn't support bind — fall back to normal invoke
            return self._invoke_llm(prompt)

        from langchain.schema import HumanMessage

        response = llm_no_think.invoke([HumanMessage(content=prompt)])
        record_llm_usage(response, process_type="reflect", function_name="pure_reflection._invoke_without_thinking")
        content = response.content if hasattr(response, 'content') else str(response)

        if self.logger and hasattr(response, 'response_metadata'):
            meta = response.response_metadata
            token_usage = meta.get('token_usage', {})
            self.logger.info(
                f"[Phase 1 Analyzer] No-think retry: "
                f"finish={meta.get('finish_reason')} "
                f"prompt_tok={token_usage.get('prompt_tokens')} "
                f"completion_tok={token_usage.get('completion_tokens')} "
                f"chars={len(content)}"
            )

        return content

    # Pattern-matching constants for "No effect"
    NO_EFFECT_PATTERNS = [
        "did not result in any changes",
        "no changes detected",
        "no changes were detected",
        "nothing happened",
        "no observable effect",
        "task did not result in",
    ]

    def _apply_no_effect_fallback(self, gradients: list, feedback_content: str) -> list:
        """Provide a fallback gradient when the LLM returns empty or all-zero gradients but feedback clearly indicates no effect"""
        # Any nonzero-magnitude gradient → do not intervene
        has_nonzero = gradients and any(g.magnitude > 0 for g in gradients)
        if has_nonzero:
            return gradients

        feedback_lower = feedback_content.lower()
        matched = any(p in feedback_lower for p in self.NO_EFFECT_PATTERNS)
        if not matched:
            return gradients

        # Case 1: empty gradient list → create a new fallback gradient (existing logic)
        if not gradients:
            direction = f"Skill executed but produced no effect. Original feedback: {feedback_content[:200]}"
            return [Gradient(
                gradient_type=GradientType.LOGIC,
                direction=direction,
                magnitude=0.4,
                evidence="Fallback: LLM returned empty gradients for a no-effect failure",
                suggested_fix="",
            )]

        # Case 2: all gradient magnitudes are zero → boost magnitudes, preserve the LLM's original analysis
        # (Gradient is a frozen dataclass, so we must create new instances)
        boosted = []
        for g in gradients:
            boosted.append(Gradient(
                gradient_type=g.gradient_type,
                magnitude=max(g.magnitude, 0.4),
                direction=g.direction,
                evidence=g.evidence + " [boosted: no-effect fallback]",
                suggested_fix=g.suggested_fix,
                affected_lines=g.affected_lines,
            ))
        if self.logger:
            self.logger.info(
                f"[Phase 1 Analyzer] No-effect fallback: boosted {len(boosted)} "
                f"zero-magnitude gradients to 0.4"
            )
        return boosted

    def _recover_self_issues_from_raw(self, response: str) -> list:
        """Extract self_issues from raw LLM response via regex.

        Called when JSON extraction fails completely (data=None) or returns
        only reasoning sub-fields (Mode E).  LLM responses frequently contain
        unescaped double quotes inside code snippets in ``suggested_fix``
        values (e.g., ``mineBlock(bot, "stone", count)``), which break
        ``json.loads`` on the matched ``self_issues`` array.  This method
        uses per-field regex extraction as a robust fallback.

        Returns:
            List of dicts suitable for gradient construction, or [].
        """
        import re as _re
        import json as _json

        if not response or '"self_issues"' not in response:
            return []

        # Step 1: try to extract and parse the self_issues array directly
        si_match = _re.search(r'"self_issues"\s*:\s*(\[.*?\])', response, _re.DOTALL)
        if si_match:
            try:
                recovered = _json.loads(si_match.group(1))
                if recovered and isinstance(recovered, list):
                    return recovered
            except (_json.JSONDecodeError, ValueError):
                pass  # Fall through to per-field extraction

        # Step 2 (Mode F): per-field regex extraction from the raw response.
        # Handles cases where code in suggested_fix has unescaped quotes
        # that corrupt json.loads.  We extract each gradient field individually.
        issues = []
        # Find all gradient_type occurrences (one per issue in self_issues)
        gt_matches = list(_re.finditer(r'"gradient_type"\s*:\s*"(\w+)"', response))
        if not gt_matches:
            return []

        for gt_m in gt_matches:
            # Search for fields AFTER this gradient_type match
            region_start = gt_m.start()
            # Region extends to the next gradient_type or end of response
            next_gt = None
            for other in gt_matches:
                if other.start() > gt_m.start():
                    next_gt = other.start()
                    break
            region = response[region_start:next_gt] if next_gt else response[region_start:]

            mag_m = _re.search(r'"magnitude"\s*:\s*([\d.]+)', region)
            dir_m = _re.search(r'"direction"\s*:\s*"([^"]+)"', region)
            ev_m = _re.search(r'"evidence"\s*:\s*"([^"]+)"', region)
            # suggested_fix is often multi-line with unescaped quotes — grab first line
            fix_m = _re.search(r'"suggested_fix"\s*:\s*"(REQUIRED:[^"]*)"', region)
            if not fix_m:
                fix_m = _re.search(r'"suggested_fix"\s*:\s*"([^"]*)"', region)

            issues.append({
                "gradient_type": gt_m.group(1),
                "magnitude": float(mag_m.group(1)) if mag_m else 0.7,
                "direction": dir_m.group(1) if dir_m else "",
                "evidence": ev_m.group(1) if ev_m else "",
                "suggested_fix": fix_m.group(1) if fix_m else "",
            })

        return issues

    def _parse_response(self, input: ReflectionInput, response: str) -> ReflectionOutput:
        """Parse the LLM response"""
        import json

        # Use the project's robust JSON extraction helper (handles markdown
        # code fences and other common LLM output wrappers).
        from ..core.llm_invoker import extract_json_from_response

        # Extract and parse JSON (handles markdown code blocks and similar formats)
        data = extract_json_from_response(
            response,
            context="Phase 1 analysis response"
        )

        # record raw response for post-mortem diagnostics (DEBUG level, doesn't affect normal log volume)
        if self.logger and data:
            self.logger.debug(
                f"[Phase 1 Analyzer] Raw response preview ({len(response)} chars): "
                f"{response[:2000]}"
            )

        if not data:
            # Before giving up, try regex recovery on the
            # raw response text.  LLM responses that contain unescaped double
            # quotes inside code snippets (e.g. mineBlock(bot, "stone", count))
            # break robust_json_parse entirely, causing data=None even though
            # the response does contain valid self_issues content.
            recovered_issues = self._recover_self_issues_from_raw(response)
            if recovered_issues:
                data = {"self_issues": recovered_issues}
                if self.logger:
                    self.logger.info(
                        f"[Phase 1 Analyzer] Mode F: recovered {len(recovered_issues)} "
                        f"self_issues from raw response (JSON extraction had failed)"
                    )
            else:
                if self.logger:
                    resp_snippet = response[:500] if response else "(empty)"
                    self.logger.warning(
                        f"[Phase 1 Analyzer] JSON extraction failed | "
                        f"skill={input.skill_name} | "
                        f"response_chars={len(response) if response else 0} | "
                        f"snippet={resp_snippet}"
                    )
                return ReflectionOutput(
                    skill_name=input.skill_name,
                    delta=SkillDelta(skill_name=input.skill_name, gradients=[]),
                    propagated_feedbacks={},
                    faulty_children=set(),
                    analysis_successful=False,
                    reasoning="Failed to parse LLM response",
                )

        # Mode G recovery — extract_json_from_response sometimes returns an
        # inner child_issues entry as if it were the top-level dict. This
        # happens when the LLM emits malformed outer JSON (unescaped braces
        # inside a string, a malformed preamble before the JSON, or bare
        # inner objects without a wrapper) so the extractor's brace counter
        # fails and its fallback regex matches one of the inner child_issues
        # objects instead of the top-level one. The result is a flat dict
        # carrying only child_issue keys, which yields zero gradients and a
        # silent optimizer no-op. Wrap the flat object back into a proper
        # child_issues list shape so the existing parsing path picks it up.
        # Must run BEFORE Mode A/D/E checks so the rest of the parser sees
        # a well-formed dict.
        if isinstance(data, dict):
            _TOP_LEVEL_KEYS = {
                "reasoning", "self_issues", "child_issues",
                "calling_pattern_issues", "parameter_corrections",
            }
            _CHILD_ISSUE_KEYS = {
                "child_skill", "issue_description", "responsibility",
                "expected_behavior", "actual_behavior", "weight",
            }
            _has_top_level = bool(_TOP_LEVEL_KEYS & set(data.keys()))
            _has_self_issue_shape = "gradient_type" in data and "direction" in data
            _child_field_count = len(_CHILD_ISSUE_KEYS & set(data.keys()))
            if (not _has_top_level and not _has_self_issue_shape
                    and _child_field_count >= 3):
                data = {"child_issues": [data]}
                if self.logger:
                    self.logger.info(
                        f"[Phase 1 Analyzer] Mode G: wrapped flat child_issue "
                        f"object (extractor returned inner dict with "
                        f"{_child_field_count}/6 child_issue keys) back into "
                        f"child_issues list"
                    )

        # Parse self_issues -> gradients
        # Fix 2A: Mode A recovery — LLM sometimes nests self_issues inside reasoning
        self_issues_data = data.get("self_issues", [])
        if not self_issues_data:
            reasoning_data = data.get("reasoning", {})
            if isinstance(reasoning_data, dict):
                nested_self = reasoning_data.get("self_issues", [])
                if nested_self:
                    self_issues_data = nested_self
                    if self.logger:
                        self.logger.info(f"[Phase 1 Analyzer] Recovered {len(nested_self)} nested self_issues from reasoning")
                    # Also recover child_issues if missing at top level
                    if not data.get("child_issues"):
                        data["child_issues"] = reasoning_data.get("child_issues", [])

        # Fix 2D: Mode D recovery — LLM put gradient fields at top level
        # instead of inside self_issues array (e.g., keys=['gradient_type','magnitude','direction',...])
        if not self_issues_data and "gradient_type" in data and "direction" in data:
            self_issues_data = [data]
            if self.logger:
                self.logger.info(
                    f"[Phase 1 Analyzer] Mode D: recovered flat gradient format "
                    f"(gradient_type={data.get('gradient_type')}, "
                    f"magnitude={data.get('magnitude')})"
                )

        # Fix 2E: Mode E recovery — JSON extractor returned reasoning
        # contents instead of full response (keys are reasoning sub-fields).
        # Re-extract self_issues from raw response text via regex.
        # Extended in Issue I to use Mode F per-field fallback when json.loads fails.
        if not self_issues_data and not data.get("child_issues"):
            reasoning_keys = {"error_understanding", "knowledge_applied", "inference", "root_cause"}
            if reasoning_keys & set(data.keys()):
                recovered_issues = self._recover_self_issues_from_raw(response)
                if recovered_issues:
                    self_issues_data = recovered_issues
                    if self.logger:
                        self.logger.info(
                            f"[Phase 1 Analyzer] Mode E+F: recovered {len(recovered_issues)} "
                            f"self_issues from raw response"
                        )

        gradients = []
        for issue in self_issues_data:
            try:
                grad_type = GradientType(issue.get("gradient_type", "logic"))
            except ValueError:
                grad_type = GradientType.LOGIC

            gradients.append(Gradient(
                gradient_type=grad_type,
                magnitude=float(issue.get("magnitude", 0.5)),
                direction=issue.get("direction", ""),
                evidence=issue.get("evidence", ""),
                suggested_fix=issue.get("suggested_fix", ""),  # Added: extract LLM-generated fix suggestion
            ))

        # "No effect" fallback: LLM returned empty gradients but feedback clearly indicates no effect
        gradients = self._apply_no_effect_fallback(gradients, input.feedback_content)

        # Fix 2C: Mode C fallback — LLM attributed all issues to children, generate degraded gradient
        if not gradients and data.get("child_issues"):
            child_issues = data["child_issues"]
            if isinstance(child_issues, list) and child_issues:
                top_child = max(child_issues, key=lambda x: float(x.get("weight", 0)))
                child_weight = float(top_child.get("weight", 0))
                if child_weight >= 0.5:
                    gradients = [Gradient(
                        gradient_type=GradientType.LOGIC,
                        direction=f"Child skill '{top_child.get('child_skill', 'unknown')}' issue: {top_child.get('issue_description', '')[:300]}",
                        magnitude=child_weight * 0.5,  # Down-weighted: halve the weight for child-skill issues
                        evidence=f"LLM attributed issue to child skill: {top_child.get('actual_behavior', '')[:200]}",
                        suggested_fix=top_child.get("issue_description", ""),
                    )]
                    if self.logger:
                        self.logger.info(
                            f"[Phase 1 Analyzer] Mode C fallback: generated gradient from child_issues "
                            f"(child={top_child.get('child_skill')}, weight={top_child.get('weight')})"
                        )

        # Parse child_issues -> propagated_feedbacks
        propagated = {}
        faulty_children = set()
        for issue in data.get("child_issues", []):
            child = issue.get("child_skill", "")
            if child and child in input.children:
                faulty_children.add(child)
                propagated[child] = PropagatedFeedback(
                    source_skill=input.skill_name,
                    target_skill=child,
                    issue_description=issue.get("issue_description", ""),
                    responsibility=issue.get("responsibility", ""),
                    expected_behavior=issue.get("expected_behavior", ""),
                    actual_behavior=issue.get("actual_behavior", "") or input.feedback_content[:300],
                    weight=float(issue.get("weight", 0.5)),
                )

        # parse calling_pattern_issues → caller-fix self-gradients
        caller_fix_children = set()
        for issue in data.get("calling_pattern_issues", []):
            child = issue.get("child_skill", "")
            if child and child in input.children:
                caller_fix_children.add(child)
                gradients.append(Gradient(
                    gradient_type=GradientType.PARAMETER_SEMANTIC,
                    direction=f"CALLER-FIX for '{child}': {issue.get('issue_description', '')}",
                    magnitude=0.8,
                    evidence=issue.get("evidence", ""),
                    suggested_fix=issue.get("suggested_fix", ""),
                ))
                if self.logger:
                    self.logger.info(
                        f"[Phase 1 Analyzer] Calling pattern issue: "
                        f"THIS skill calls '{child}' incorrectly — {issue.get('issue_description', '')[:200]}"
                    )

        # Parse parameter_corrections from Phase 1 RCA
        parameter_corrections = []
        for correction in data.get("parameter_corrections", []):
            param = correction.get("param_name")
            if param and "passed_value" in correction and "suggested_value" in correction:
                conf = float(correction.get("confidence", 0.7))
                if conf >= 0.7:  # Confidence threshold
                    parameter_corrections.append({
                        # target_skill: LLM's attribution of which skill's parameter
                        # was wrong. May be None for responses from before the schema
                        # update; reflection_chain falls back to the reflected node
                        # name in that case.
                        "target_skill": correction.get("target_skill"),
                        "param_name": param,
                        "passed_value": correction["passed_value"],
                        "suggested_value": correction["suggested_value"],
                        "reason": correction.get("reason", ""),
                        "confidence": conf,
                    })
                    if self.logger:
                        target_str = correction.get("target_skill") or "(unattributed)"
                        self.logger.info(
                            f"[Phase 1 Analyzer] Parameter correction: "
                            f"{target_str}.{param}={correction['passed_value']!r} → "
                            f"{correction['suggested_value']!r} (conf={conf:.1f})"
                        )

        # Parse the reasoning chain (explicit reasoning chain support)
        reasoning_data = data.get("reasoning", {})
        reasoning_chain = None

        # If reasoning is a structured object (new format)
        if isinstance(reasoning_data, dict):
            reasoning_chain = ReasoningChain(
                retrieved_knowledge=getattr(self, '_retrieved_knowledge', []),
                error_understanding=reasoning_data.get("error_understanding", ""),
                knowledge_applied=reasoning_data.get("knowledge_applied", []),
                inference=reasoning_data.get("inference", ""),
                raw_text=json.dumps(reasoning_data),
            )
            reasoning_str = reasoning_data.get("inference", "") or json.dumps(reasoning_data)
        else:
            # Backward compatibility: if reasoning is a string (old format)
            reasoning_str = str(reasoning_data)
            # Still record retrieved knowledge
            if hasattr(self, '_retrieved_knowledge') and self._retrieved_knowledge:
                reasoning_chain = ReasoningChain(
                    retrieved_knowledge=self._retrieved_knowledge,
                    raw_text=reasoning_str,
                )

        # Fix 2G: Log zero gradient warning for diagnostics
        if data and not gradients and self.logger:
            self.logger.warning(
                f"[Phase 1 Analyzer] Zero gradients despite successful JSON extraction | "
                f"keys={list(data.keys())} | "
                f"self_issues={len(data.get('self_issues', []))} | "
                f"child_issues={len(data.get('child_issues', []))} | "
                f"response_chars={len(response)}"
            )

        # Actionability gate. Previously analysis_successful was set to True
        # unconditionally whenever data was non-None — so when the JSON
        # extractor salvaged only a fragment like {"confidence": 0.X} from a
        # malformed response, _parse_response would still report success → no
        # gradients constructed → the repair-prompt retry path didn't fire →
        # the optimizer silently no-op'd that cycle. Gate instead on the same
        # actionability signal the downstream consumers care about: non-empty
        # gradients, propagated child feedbacks, or parameter corrections.
        _is_actionable = bool(gradients or propagated or parameter_corrections)
        return ReflectionOutput(
            skill_name=input.skill_name,
            delta=SkillDelta(skill_name=input.skill_name, gradients=gradients),
            propagated_feedbacks=propagated,
            faulty_children=faulty_children,
            analysis_successful=_is_actionable,
            reasoning=reasoning_str,
            reasoning_chain=reasoning_chain,
            caller_fix_children=caller_fix_children,
            parameter_corrections=parameter_corrections,
        )


# ============================================================
# Part 5: convenience functions and factories
# ============================================================

def create_pure_reflection(
    llm=None,
    logger=None,
    mode: str = "llm",
    pure_reasoning: bool = False,
    include_reasoning_examples: bool = False,
    use_factual_primitive_doc: bool = False,
    **kwargs,
) -> PureReflection:
    """
    Create a PureReflection instance

    Args:
        llm: LLM instance
        logger: logger
        mode: analysis mode (only "llm" is supported)
        pure_reasoning: strip domain data injection (conclusions, etc.); used for evaluation
        include_reasoning_examples: gate the reasoning_examples_section build in Phase 1.
            Default False per a cross-LLM ablation study.
        use_factual_primitive_doc: gate factual primitive-doc injection
            in Phase 1. Default False; when True, doc strings for primitives
            detected in skill_code are appended to the analysis prompt.

    Returns:
        PureReflection: configured reflector
    """
    if not llm:
        raise ValueError("LLM instance is required for PureReflection")

    analyzer = LLMAnalyzer(
        llm,
        logger,
        pure_reasoning=pure_reasoning,
        include_reasoning_examples=include_reasoning_examples,
        use_factual_primitive_doc=use_factual_primitive_doc,
    )
    return PureReflection(analyzer)
