"""Minecraft action space.

Houses the runtime bridge (``env/``), the JavaScript control primitives
(``control_primitives/``), and the LLM prompt context overlays for them
(``control_primitives_context*/``). These were previously rooted at the
``skillnet/`` top level; they are now part of the Minecraft domain.

Top-level shim modules (``skillnet.env``, ``skillnet.control_primitives``,
``skillnet.control_primitives_context``) re-export from here so existing
imports continue to work.
"""
