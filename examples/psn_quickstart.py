"""PSN Quickstart Example — smallest possible PSN run.

Smoke-tests the agent end to end with a tiny budget (5 iterations) and
optimization disabled, so the run finishes in minutes rather than hours.
This is NOT the recommended setup for production use; for the full
learning loop see ``PSNConfig.production()`` or just run
``./skillnet.sh``.

Prerequisites (see README):
  1. ``make install``                    — Python + Node dependencies
  2. ``.env`` with OPENAI_API_KEY (or vLLM config)
  3. EITHER an open-to-LAN Minecraft client (Path A) OR a running
     headless Fabric server (Path B via ``./skillnet.sh --mc-mode=headless``)

PSN auto-detects the Minecraft port on localhost via the real
Minecraft handshake, so this script does not hard-code a port. Pass
``mc_port=PORT`` to ``from_registered_domain`` if you need to be explicit.
"""

from skillnet import PSNAgent
from skillnet.core.config import PSNConfig


def main():
    # Tiny config: 5 iterations, debug on, optimizer/refactor disabled.
    # See PSNConfig.production() for the full learning loop config.
    config = PSNConfig.quick_test(max_iterations=5)

    # Build the agent against the registered Minecraft domain. PSN scans
    # local Java ports for an open-to-LAN Minecraft handshake. To pin
    # an explicit port (e.g. when multiple Minecraft servers are
    # running), pass `mc_port=39393` below.
    agent = PSNAgent.from_registered_domain("minecraft", config=config)

    result = agent.learn()
    print(f"completed_tasks: {result['completed_tasks']}")
    print(f"failed_tasks:    {result['failed_tasks']}")
    print(f"learned skills:  {len(result['skills'])}")


if __name__ == "__main__":
    main()
