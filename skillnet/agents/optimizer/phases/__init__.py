"""
Optimizer phases — two-phase optimization pipeline.

Core flow:
1. Top-Down analysis (``PureReflection``): start from the failing skill,
   recursively analyze issues, produce deltas.
2. Bottom-Up optimization (``ConsiderFunction``): optimize upward from leaf
   nodes, propagating forward feedbacks.

Submodules (import directly):
- ``pure_reflection``      PureReflection, SkillDelta, Gradient, ...
- ``consider_function``    ConsiderFunction, OptimizationForwardFeedback, ...
- ``reflection_chain``     ReflectionChain, ChainExecutor,
                           TwoPhaseOptimizationPipeline, create_two_phase_pipeline
- ``skill_info_adapter``   SkillInfo, SkillInfoGetter, create_skill_info_getter
- ``env_knowledge``        environment-knowledge rendering
- ``factual_primitive_doc`` primitive-doc rendering
"""
