"""
Skill Graph Optimizer Package

Skill graph optimizer module, implementing a two-phase optimization workflow.

Main classes:
- SkillGraphOptimizer: original optimizer implementation (from the old skill_graph_optimizer.py)
- TwoPhaseOptimizationEngine: new two-phase optimization engine

All symbols are available from their respective subpackages:
- ._impl: SkillGraphOptimizer, OptimizationTracker, CodeBloatTracker
- .engine: TwoPhaseOptimizationEngine, TwoPhaseOptimizationConfig
- .phases: PureReflection, ReflectionChain, ChainExecutor, etc.
- .feedback: OptimizationForwardFeedback, SkillFeedback, etc.
- .tracking: TransactionManager, etc.
- .validators: BloatChecker, SkipChecker, EffectsConsistencyValidator, etc.
- .config: BloatPreventionConfig, SkipOptimizationConfig, etc.
"""
