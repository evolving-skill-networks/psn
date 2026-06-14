import psutil
from skillnet import PSNAgent
from dotenv import load_dotenv
import os
import socket
import struct
import json
import argparse


def to_varint(n):
    """Convert integer to VarInt (used in Minecraft networking)."""
    byte_array = bytearray()
    while (n & ~0x7F) != 0:
        byte_array.append((n & 0x7F) | 0x80)
        n >>= 7
    byte_array.append(n)
    return bytes(byte_array)


def verify_minecraft_server(ip_addr, port_number):
    sock = None
    try:
        # Connect to the server with a timeout
        sock = socket.create_connection((ip_addr, port_number), timeout=5)

        # Construct the Handshake packet
        # 1. Packet ID (1 byte, for Handshake this is 0x00)
        packet_id = b"\x00"
        # 2. Protocol Version (as of 1.19, the protocol version is 759; adjust if necessary)
        protocol_version = to_varint(759)
        # 3. Length of IP address followed by actual IP address
        ip_address_bytes = ip_addr.encode("utf-8")
        ip_address_length = to_varint(len(ip_address_bytes))
        # 4. Port number (2 bytes in big-endian format)
        port_bytes = struct.pack("!H", port_number)
        # 5. Next state (1 byte, 0x01 for status)
        next_state = b"\x01"

        # Combine all parts into a Handshake packet
        handshake_packet = (
                packet_id
                + protocol_version
                + ip_address_length
                + ip_address_bytes
                + port_bytes
                + next_state
        )

        # Prepend the length of the packet as a VarInt
        handshake_length = to_varint(len(handshake_packet))
        full_handshake_packet = handshake_length + handshake_packet

        # Send the Handshake packet
        sock.sendall(full_handshake_packet)

        # Send the Status Request packet (1 byte, 0x00)
        status_request_packet = b"\x01\x00"  # Length of 1 (VarInt) + 0x00 (Packet ID)
        sock.sendall(status_request_packet)

        # Read the response
        # Read the length of the packet (VarInt)
        to_varint(read_varint(sock))
        # Read the Packet ID (VarInt, not needed further)
        read_varint(sock)
        # Read the JSON Data (as a string)
        json_length = read_varint(sock)  # Length of the JSON string
        response = sock.recv(json_length).decode("utf-8")

        # Parse and verify the JSON response
        server_data = json.loads(response)

        if "description" in server_data and "players" in server_data:
            #print(f"Found Minecraft server on port {port_number}")
            return True
        else:
            #print(f"Port {port_number} is not a valid Minecraft server.")
            return False

    except Exception as e:
        #print(f"Exception on port {port_number}: {e}")
        return False

    finally:
        if sock is not None:
            sock.close()


def read_varint(sock):
    """Reads a VarInt from the socket."""
    value = 0
    size = 0
    while True:
        byte = sock.recv(1)
        if len(byte) == 0:
            raise IOError("Unexpected end of stream while reading VarInt")
        byte = ord(byte) if isinstance(byte, bytes) else byte  # Extract int from byte
        value |= (byte & 0x7F) << (7 * size)
        size += 1
        if size > 5:
            raise IOError("VarInt is too big.")
        if not (byte & 0x80):
            break
    return value

# Known auxiliary java ports that LISTEN but are not Minecraft servers.
# 24464 = ServerReplay viewer-pack port; co-listens alongside a fabric server.
_MC_PORT_SKIP = {24464}


def find_minecraft_java_port():
    """
    Finds a port on which a Minecraft Java server is listening.

    Probes every listening Java socket with the real Minecraft status-handshake
    (verify_minecraft_server) and returns the first port that responds with a
    valid MC status JSON. The earlier heuristic of grep'ing each process's
    cmdline for the substring 'minecraft' missed standalone fabric server
    launchers (e.g. `java -jar fabric-server-launcher.jar nogui`, whose cmdline
    has no such string) — those setups silently auto-detected nothing or
    fell through to a co-listening Modrinth/launcher client on a different
    port. Probing via the handshake is both more inclusive and stricter.

    If MORE than one valid MC server is found, logs all candidates and tells
    the caller to pass `--mc-port` explicitly. The choice of which one to
    return in that case is the lowest port number (deterministic for tests),
    but the warning makes the ambiguity visible.

    Returns:
        int | None: a responding MC port, or None if no Java listener
        answered the handshake.
    """
    java_ports = set()
    for conn in psutil.net_connections(kind='inet'):
        if conn.status != psutil.CONN_LISTEN:
            continue
        if conn.laddr.port in _MC_PORT_SKIP:
            continue
        try:
            process = psutil.Process(conn.pid)
            if 'java' in process.name().lower():
                java_ports.add(conn.laddr.port)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    mc_ports = [p for p in sorted(java_ports) if verify_minecraft_server("127.0.0.1", p)]
    if not mc_ports:
        return None
    if len(mc_ports) > 1:
        print(
            f"[mc] WARNING: multiple Minecraft servers detected on 127.0.0.1: {mc_ports}. "
            f"Auto-detect picked {mc_ports[0]}; pass --mc-port=<port> to choose explicitly."
        )
    return mc_ports[0]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SkillNet - An Open-Ended Embodied Agent')

    # Run mode options
    parser.add_argument('--manual', action='store_true',
                        help='Use manual curriculum mode (you input tasks)')
    parser.add_argument('--inference', action='store_true',
                        help='Run in inference mode (single task) instead of learn mode')

    # Planner mode
    parser.add_argument('--planner-mode', choices=['graph', 'adaptive'], default=None,
                        help='Planner mode: graph (always use graph planning) or adaptive (default)')

    # Optimizer/Refactor options
    parser.add_argument('--no-optimizer', action='store_true',
                        help='Disable skill optimizer (only applies to graph mode)')
    parser.add_argument('--no-refactor', action='store_true',
                        help='Disable skill refactoring (only applies to graph mode)')
    parser.add_argument('--postmilestone-adaptive', action='store_true',
                        help='After all milestones, keep the adaptive learning-path '
                             'continuation/decomposition (old behavior). Default: each '
                             'iteration runs a fresh LLM-proposed task for '
                             'open-exploration breadth')
    parser.add_argument('--include-skill-code', action='store_true',
                        help='Include full skill code in action agent prompt (default: signatures only)')
    parser.add_argument('--pure-reasoning', action='store_true',
                        help='Remove domain-specific data injection from optimization prompts (for evaluation)')
    parser.add_argument('--combat', action='store_true',
                        help='Combat mode (Minecraft): do not force peaceful/day on reset, so hostile '
                             'mobs spawn and the manually-set difficulty/time persist')

    # LLM backend selection
    parser.add_argument('--model', choices=['openai', 'vllm'], default=None,
                        help='LLM backend: openai (gpt-5-mini via OpenAI API) or vllm (from .env)')

    # Domain selection (defaults to minecraft; dynamic choices reflect registered domains)
    from skillnet.domains import list_domains
    parser.add_argument('--domain', choices=list_domains(), default='minecraft',
                        help='Which domain to run (default: minecraft). Choices reflect '
                             'currently-registered domains in skillnet.domains.')

    # Checkpoint directory
    parser.add_argument('--ckpt-dir', type=str, default='ckpt',
                        help='Checkpoint directory (default: ckpt)')

    # Explicit Minecraft server port (skips auto-detect)
    parser.add_argument('--mc-port', type=int, default=None,
                        help='Explicit Minecraft server port on localhost. '
                             'Skips auto-detect via find_minecraft_java_port(). '
                             'Use to point at a dedicated Fabric server (e.g. 25565) '
                             'instead of the integrated client LAN.')

    # Experiment parameters
    parser.add_argument('--optimization-threshold', type=float, default=None,
                        help='Maturity gating pivot theta (default: 0.6)')
    parser.add_argument('--optimization-epsilon', type=float, default=None,
                        help='Maturity gating floor probability epsilon (default: 0.05)')
    parser.add_argument('--optimization-gamma', type=float, default=None,
                        help='Maturity gating sigmoid slope gamma (default: 8.0)')
    parser.add_argument('--max-skills', type=int, default=None,
                        help='Skill graph capacity cap (default: unlimited)')
    parser.add_argument('--graph-snapshot-interval', type=int, default=None,
                        help='Graph snapshot interval in iterations (0=disabled)')
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Max learning iterations (default: 3160)')

    # Non-interactive flags (for automated sweeps)
    parser.add_argument('--resume', action='store_true', default=None, dest='resume',
                        help='Resume from previous session (skip prompt)')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                        help='Start fresh (skip prompt)')
    parser.add_argument('--debug', action='store_true', default=None, dest='debug',
                        help='Enable debug mode (skip prompt)')
    parser.add_argument('--no-debug', action='store_false', dest='debug',
                        help='Disable debug mode (skip prompt)')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    print("        Evolving Programmatic Skill Networks (PSN)")
    print("    Lifelong-learning LLM agents with composable, executable skills")
    print()
    print("                authors: Haochen Shi, Xingdi Yuan, Bang Liu (2026)")
    print("                project: https://evolving-skill-networks.github.io/")
    print()

    # Determine mode configuration
    planner_mode = 'adaptive'
    if args.planner_mode:
        planner_mode = args.planner_mode

    curriculum_agent_mode = "manual" if args.manual else "auto"
    run_mode = "inference" if args.inference else "learn"

    print(f"  Planner: {planner_mode}")
    print(f"  Optimizer: {'disabled' if args.no_optimizer else 'enabled'}")
    print(f"  Refactor: {'disabled' if args.no_refactor else 'enabled'}")
    print(f"  Post-milestone adaptive path: {'enabled' if args.postmilestone_adaptive else 'disabled'}")
    print(f"  Pure reasoning: {'enabled' if args.pure_reasoning else 'disabled'}")
    print(f"  Curriculum agent mode: {curriculum_agent_mode}")
    print(f"  Run mode: {run_mode}")
    print(f"  Checkpoint dir: {args.ckpt_dir}")
    if args.optimization_threshold is not None:
        print(f"  Optimization threshold (θ): {args.optimization_threshold}")
    if args.optimization_epsilon is not None:
        print(f"  Optimization epsilon (ε): {args.optimization_epsilon}")
    if args.optimization_gamma is not None:
        print(f"  Optimization gamma (γ): {args.optimization_gamma}")
    if args.max_skills is not None:
        print(f"  Max skills: {args.max_skills}")
    if args.graph_snapshot_interval is not None:
        print(f"  Graph snapshot interval: {args.graph_snapshot_interval}")
    print()

    # Load environment variables from .env file.
    # override=True ensures .env values take priority over any pre-existing
    # shell/conda environment variables (prevents stale keys from a user's
    # global shell profile silently shadowing the project .env).
    load_dotenv(override=True)

    # Override LLM backend based on --model flag
    if args.model == 'openai':
        os.environ.pop('VLLM_API_BASE', None)
        os.environ.pop('VLLM_API_KEY', None)
        os.environ.pop('VLLM_MODEL', None)
        if not os.getenv('OPENAI_MODEL'):
            os.environ['OPENAI_MODEL'] = 'gpt-5-mini'
        print(f"\033[32m[Model] Using OpenAI API ({os.environ['OPENAI_MODEL']})\033[0m")
    elif args.model == 'vllm':
        print(f"\033[32m[Model] Using vLLM backend (from .env)\033[0m")

    # Apply RobustnessConfig from environment variables
    from skillnet.config.robustness_config import RobustnessConfig
    RobustnessConfig.from_env()

    # Find the port number for the Minecraft Java server.
    # If --mc-port is given, skip auto-detect and use it directly — useful when
    # pointing at a dedicated Fabric server alongside a running client.
    if getattr(args, 'mc_port', None) is not None:
        mc_port = args.mc_port
        print(f"\033[36m[mc] Using explicit --mc-port={mc_port} (auto-detect skipped)\033[0m")
    else:
        mc_port = find_minecraft_java_port()

    # Get OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")

    # vLLM / OpenAI-compatible backend configuration
    vllm_api_base = os.getenv("VLLM_API_BASE")
    vllm_api_key = os.getenv("VLLM_API_KEY")
    vllm_model = os.getenv("VLLM_MODEL")

    # Validate vLLM endpoint reachability before proceeding
    if vllm_api_base:
        import sys
        import urllib.request
        health_url = vllm_api_base.replace("/v1", "/health")
        try:
            urllib.request.urlopen(health_url, timeout=10)
            print(f"\033[32m[vLLM] Endpoint healthy: {vllm_api_base}\033[0m")
        except Exception as e:
            print(f"\033[31m[vLLM] ERROR: Endpoint unreachable: {health_url}\033[0m")
            print(f"\033[31m[vLLM] {e}\033[0m")
            print(f"\033[33m[vLLM] Please check: 1) EC2 instance running  2) .env IP correct  3) Security group allows access\033[0m")
            sys.exit(1)

    if mc_port is None:
        print()
        print("\033[41;33m**************************************************************************\033[0m")
        print("\033[41;33m*** Unable to detect a Minecraft Java server on localhost.             ***\033[0m")
        print("\033[41;33m***                                                                    ***\033[0m")
        print("\033[41;33m*** Two supported setups:                                              ***\033[0m")
        print("\033[41;33m***   A) Minecraft client (Modrinth launcher recommended)              ***\033[0m")
        print("\033[41;33m***      - Open a singleplayer world with Fabric loader installed     ***\033[0m")
        print("\033[41;33m***      - Click 'Open to LAN' and allow cheats                       ***\033[0m")
        print("\033[41;33m***      - Re-run ./skillnet.sh                                       ***\033[0m")
        print("\033[41;33m***                                                                    ***\033[0m")
        print("\033[41;33m***   B) Headless Fabric server (no GUI required)                     ***\033[0m")
        print("\033[41;33m***      - ./skillnet.sh --mc-mode=headless                           ***\033[0m")
        print("\033[41;33m***        (PSN sets up + manages the server for you)                 ***\033[0m")
        print("\033[41;33m***                                                                    ***\033[0m")
        print("\033[41;33m*** Or pass --mc-port=PORT explicitly if you know the port.            ***\033[0m")
        print("\033[41;33m**************************************************************************\033[0m")
        print()
        exit()

    # Resume: CLI flag or interactive prompt
    if args.resume is not None:
        resume = args.resume
    else:
        resume = input("Do you want to continue from your previous session? (yes/no): ").strip().lower().startswith('y')

    # Debug: CLI flag or interactive prompt
    if args.debug is not None:
        debug_mode = args.debug
    else:
        debug_mode = input("Enable debug mode? (yes/no): ").strip().lower().startswith('y')

    # Per-component model configuration
    # Priority: per-component env var > global vLLM/OpenAI model
    # API endpoints are resolved automatically by create_chat_llm()
    default_model = vllm_model or openai_model

    action_agent_model = os.getenv("ACTION_AGENT_MODEL", default_model)
    curriculum_agent_model = os.getenv("CURRICULUM_AGENT_MODEL", default_model)
    curriculum_agent_qa_model = os.getenv("CURRICULUM_AGENT_QA_MODEL", default_model)
    critic_agent_model = os.getenv("CRITIC_AGENT_MODEL", default_model)
    skill_manager_model = os.getenv("SKILL_MANAGER_MODEL", default_model)

    # Set global API key as environment variable
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    # Build structured PSNConfig from CLI args and environment
    from skillnet.core.config import (
        PSNConfig, AgentLLMs, LLMEndpoint,
        ActionConfig, CurriculumConfig, PlannerConfig,
        OptimizationConfig, CheckpointConfig, RecordingConfig,
        SkillManagerConfig,
    )

    # Build OptimizationConfig, only overriding when explicitly set
    opt_kwargs = {
        'enable_optimizer': not args.no_optimizer,
        'enable_refactor': not args.no_refactor,
        'pure_reasoning': args.pure_reasoning,
    }
    if args.optimization_threshold is not None:
        opt_kwargs['optimization_threshold'] = args.optimization_threshold
    if args.optimization_epsilon is not None:
        opt_kwargs['optimization_epsilon'] = args.optimization_epsilon
    if args.optimization_gamma is not None:
        opt_kwargs['optimization_gamma'] = args.optimization_gamma

    # Build SkillManagerConfig
    sm_kwargs = {}
    if args.max_skills is not None:
        sm_kwargs['max_skills'] = args.max_skills

    # Build CheckpointConfig
    ckpt_kwargs = {'ckpt_dir': args.ckpt_dir, 'resume': resume}
    if args.graph_snapshot_interval is not None:
        ckpt_kwargs['graph_snapshot_interval'] = args.graph_snapshot_interval

    # Top-level PSNConfig overrides
    psn_kwargs = {}
    if args.max_iterations is not None:
        psn_kwargs['max_iterations'] = args.max_iterations

    # Model-aware defaults via ModelProfile
    from skillnet.core.model_profile import detect_model_profile
    _profile = detect_model_profile(action_agent_model or "")
    # CLI flag overrides profile default
    _include_skill_code = args.include_skill_code if args.include_skill_code else _profile.include_skill_code
    print(
        f"\033[36m[ModelProfile] {_profile.name}: "
        f"include_skill_code={_include_skill_code}, "
        f"thinking={_profile.enable_thinking}, "
        f"temp={_profile.fixed_temperature}\033[0m"
    )

    config = PSNConfig(
        llm=AgentLLMs(
            action=LLMEndpoint(model_name=action_agent_model),
            curriculum=LLMEndpoint(model_name=curriculum_agent_model),
            curriculum_qa=LLMEndpoint(model_name=curriculum_agent_qa_model),
            critic=LLMEndpoint(model_name=critic_agent_model),
            skill_manager=LLMEndpoint(model_name=skill_manager_model),
        ),
        action=ActionConfig(include_skill_code=_include_skill_code),
        curriculum=CurriculumConfig(
            mode=curriculum_agent_mode,
            postmilestone_adaptive_path=args.postmilestone_adaptive,
        ),
        planner=PlannerConfig(mode=planner_mode),
        optimization=OptimizationConfig(**opt_kwargs),
        skill_manager=SkillManagerConfig(**sm_kwargs),
        checkpoint=CheckpointConfig(**ckpt_kwargs),
        recording=RecordingConfig(),
        debug_mode=debug_mode,
        **psn_kwargs,
    )

    # Build domain and create agent via from_domain path
    # Create LLM for knowledge retrieval
    from skillnet.utils.llm_factory import create_chat_llm
    kr_llm = create_chat_llm(
        model_name=action_agent_model,
        temperature=0,
        request_timeout=60,
        component="optimizer",
    )
    # Domain dispatch via the registry. Minecraft passes its MC-specific kwargs;
    # other domains receive only the LLM-agnostic kwargs (model_name + kr_llm).
    # When a domain needs its own runtime kwargs, those flags should be added to
    # argparse separately and forwarded here.
    from skillnet.domains import get_domain
    if args.domain == 'minecraft':
        domain = get_domain('minecraft',
            mc_port=mc_port, model_name=action_agent_model, kr_llm=kr_llm,
            combat=args.combat)
    else:
        if args.mc_port is not None:
            raise SystemExit(
                "--mc-port is only valid with --domain=minecraft "
                f"(got --domain={args.domain})"
            )
        domain = get_domain(args.domain, model_name=action_agent_model, kr_llm=kr_llm)
    agent = PSNAgent.from_domain(domain, config)

    # Run in the selected mode
    if args.inference:
        # Inference mode: run single task
        if args.manual:
            # Manual inference: ask user for task
            task = input("Enter task to execute: ").strip()
            if task:
                print(f"Running inference for task: {task}")
                agent.inference(task=task)
            else:
                print("No task provided, exiting.")
        else:
            # Auto inference: let curriculum agent propose a task
            print("Running inference with auto-proposed task...")
            agent.inference()
    else:
        # Learn mode: continuous learning loop. Wrap in try/finally so we
        # ALWAYS terminate the mineflayer Node subprocess + close the MC env
        # on exit (graceful or via Ctrl-C / max-iterations / exception).
        # Without this, the SubprocessMonitor thread kept Python alive
        # (we daemonized the thread; this finally also frees the port + bot
        # connection so the next run boots cleanly).
        try:
            agent.learn()
        finally:
            try:
                agent.env.close()
            except Exception as e:
                print(f"[exit] env.close error: {e}")