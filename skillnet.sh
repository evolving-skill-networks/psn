#!/bin/bash
set -e

# Load nvm if available
export NVM_DIR="$HOME/.nvm"
if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    source "$NVM_DIR/nvm.sh"
elif [[ -s "/usr/local/opt/nvm/nvm.sh" ]]; then
    source "/usr/local/opt/nvm/nvm.sh"
fi

# Function to setup Node.js version
setup_node_version() {
    # Check if nvm is available (nvm is a function, not a command)
    if type nvm &>/dev/null || [[ -n "$NVM_DIR" ]]; then
        # Try to use Node.js 20.19.5 if available
        if nvm list 2>/dev/null | grep -q "v20.19.5"; then
            echo "Switching to Node.js 20.19.5..."
            nvm use 20.19.5 2>/dev/null || nvm use v20.19.5 2>/dev/null || true
        elif nvm list 2>/dev/null | grep -q "20"; then
            # Use any Node.js 20.x version if available
            NODE_20_VERSION=$(nvm list 2>/dev/null | grep "v20" | head -1 | awk '{print $1}' | tr -d '->* ')
            if [[ -n "$NODE_20_VERSION" ]]; then
                echo "Switching to Node.js $NODE_20_VERSION..."
                nvm use "$NODE_20_VERSION" 2>/dev/null || true
            fi
        fi
    fi
}

# Setup Node.js version early
setup_node_version

# Verbose flag (will be set by argument parsing)
VERBOSE=""
NPM_QUIET="--quiet"

# Function to show usage
show_usage() {
    echo "Usage: skillnet.sh [install] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  install      Install or reinstall the application"
    echo ""
    echo "Planner Mode:"
    echo "  --planner-mode=MODE          graph (always use graph planning) or adaptive (default)"
    echo ""
    echo "Feature Toggles (graph mode only):"
    echo "  --no-optimizer   Disable skill optimizer"
    echo "  --no-refactor    Disable skill refactoring"
    echo "  --postmilestone-adaptive  Post-milestone: keep the adaptive learning-path (default: one fresh LLM task per iteration)"
    echo "  --include-skill-code  Include full skill code in action agent prompt (default: signatures only)"
    echo "  --pure-reasoning   Remove domain data injection from optimization (for evaluation)"
    echo "  --combat         Minecraft: don't force peaceful/day on reset (mobs spawn; keep manual difficulty/time)"
    echo ""
    echo "LLM Backend:"
    echo "  --model=openai  Use OpenAI API (gpt-5-mini)"
    echo "  --model=vllm    Use vLLM backend (from .env) [default when .env has VLLM config]"
    echo ""
    echo "Domain Selection:"
    echo "  --domain=NAME   Which domain to run. Default: minecraft."
    echo "                  Choices are dynamic — currently registered domains"
    echo "                  appear in 'skillnet --help'."
    echo ""
    echo "Experiment Parameters:"
    echo "  --optimization-threshold=FLOAT  Maturity gating theta (default: 0.6)"
    echo "  --optimization-epsilon=FLOAT    Maturity gating epsilon (default: 0.05)"
    echo "  --optimization-gamma=FLOAT      Maturity gating gamma (default: 8.0)"
    echo "  --max-skills=INT                Skill graph capacity cap (default: unlimited)"
    echo "  --graph-snapshot-interval=INT   Graph snapshot interval (0=disabled)"
    echo "  --max-iterations=INT            Max learning iterations (default: 3160)"
    echo "  --resume / --no-resume          Resume previous session (skip prompt)"
    echo "  --debug / --no-debug            Enable debug mode (skip prompt)"
    echo ""
    echo "Run Options:"
    echo "  --manual        Use manual curriculum mode (you input tasks)"
    echo "  --auto          Use automatic curriculum mode (default)"
    echo "  --inference     Run single task inference mode"
    echo "  --learn         Run continuous learning mode (default)"
    echo "  --ckpt-dir=DIR  Checkpoint directory (default: ckpt)"
    echo "  --verbose       Show detailed output"
    echo ""
    echo "Minecraft Setup:"
    echo "  --mc-mode=lan       Connect to an open-to-LAN Minecraft client (default;"
    echo "                      we recommend launching the client via Modrinth). PSN"
    echo "                      auto-scans local Java ports for the MC handshake."
    echo "  --mc-mode=headless  Set up + manage a headless Fabric server (no GUI"
    echo "                      required). Config:"
    echo "                      installation/fabric_server_config.json"
    echo "  --mc-port=PORT      Advanced: explicit port override; skips auto-detect."
    echo ""
    echo "Examples:"
    echo "  skillnet.sh                       # Run against an open-to-LAN Minecraft client"
    echo "  skillnet.sh --mc-mode=headless    # Run with managed headless Fabric server"
    echo "  skillnet.sh --no-optimizer        # Run PSN without the skill optimizer"
    echo "  skillnet.sh --manual              # Run with manual task input"
    echo "  skillnet.sh --domain=minecraft    # Run the Minecraft domain (default; explicit form)"
    echo "  skillnet.sh install               # Install dependencies"
    exit 1
}

# Function to select checkpoint directory interactively
select_checkpoint() {
    # Find all ckpt* directories in current directory
    local ckpt_dirs=()
    while IFS= read -r -d '' dir; do
        # Only include directories that look like checkpoints (have skill_graph or progress subdirs)
        if [[ -d "$dir/skill_graph" ]] || [[ -d "$dir/progress" ]]; then
            ckpt_dirs+=("$dir")
        fi
    done < <(find . -maxdepth 1 -type d -name "ckpt*" -print0 2>/dev/null | sort -z)

    # If no checkpoint directories found
    if [[ ${#ckpt_dirs[@]} -eq 0 ]]; then
        echo "No existing checkpoint directories found. Will use default 'ckpt'."
        CKPT_DIR="ckpt"
        return
    fi

    # Let user choose even if there's only one checkpoint
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────────┐"
    echo "│                     Available Checkpoint Directories                     │"
    echo "├────┬────────────────────────────────────────────────┬─────────┬─────────┤"
    echo "│ #  │ Directory                                      │ Skills  │ Iters   │"
    echo "├────┼────────────────────────────────────────────────┼─────────┼─────────┤"

    local i=1
    for dir in "${ckpt_dirs[@]}"; do
        local dir_name="${dir#./}"
        local skill_count=0
        local iteration_count=0

        # Count skills if skill_graph/code exists
        if [[ -d "$dir/skill_graph/code" ]]; then
            skill_count=$(ls -1 "$dir/skill_graph/code"/*.js 2>/dev/null | wc -l)
        fi

        # Count iterations if iteration_log exists
        if [[ -f "$dir/progress/iteration_log.jsonl" ]]; then
            iteration_count=$(wc -l < "$dir/progress/iteration_log.jsonl" 2>/dev/null || echo 0)
        fi

        # Truncate long directory names
        local display_name="$dir_name"
        if [[ ${#dir_name} -gt 46 ]]; then
            display_name="${dir_name:0:43}..."
        fi

        printf "│ %2d │ %-46s │ %7d │ %7d │\n" "$i" "$display_name" "$skill_count" "$iteration_count"
        ((i++))
    done

    echo "├────┼────────────────────────────────────────────────┼─────────┼─────────┤"
    printf "│  0 │ %-46s │    new  │    new  │\n" "Create new checkpoint (ckpt)"
    echo "└────┴────────────────────────────────────────────────┴─────────┴─────────┘"
    echo ""

    # Prompt for selection
    while true; do
        read -p "Select checkpoint [0-$((${#ckpt_dirs[@]}))]: " selection

        # Validate input
        if [[ "$selection" =~ ^[0-9]+$ ]]; then
            if [[ "$selection" -eq 0 ]]; then
                CKPT_DIR="ckpt"
                echo "Using new checkpoint: $CKPT_DIR"
                return
            elif [[ "$selection" -ge 1 ]] && [[ "$selection" -le ${#ckpt_dirs[@]} ]]; then
                CKPT_DIR="${ckpt_dirs[$((selection-1))]#./}"
                echo "Selected checkpoint: $CKPT_DIR"
                return
            fi
        fi

        echo "Invalid selection. Please enter a number between 0 and ${#ckpt_dirs[@]}."
    done
}

# Function to run the application
run_app() {
    # Check if conda environment is active
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        echo ""
        echo "Warning: No conda environment detected. Please activate your conda environment first."
        echo "Example: conda activate <your_env_name>"
        echo ""
    else
        echo "Using conda environment: $CONDA_DEFAULT_ENV"
    fi

    # If no checkpoint directory specified, offer interactive selection
    if [[ -z "$CKPT_DIR" ]]; then
        select_checkpoint
    fi

    # ============================================================
    # Minecraft server mode handling
    # ============================================================
    # Default mode is "lan" — PSN auto-detects an open-to-LAN client.
    # "headless" mode means we set up + manage a Fabric server here.
    # "--mc-port=PORT" is an advanced manual override that bypasses both.
    MC_MODE="${MC_MODE:-lan}"
    if [[ "$MC_MODE" != "lan" && "$MC_MODE" != "headless" ]]; then
        echo "ERROR: --mc-mode must be 'lan' or 'headless' (got '$MC_MODE')" >&2
        exit 1
    fi

    if [[ -n "$MC_PORT_OVERRIDE" && "$MC_MODE" == "headless" ]]; then
        echo "ERROR: --mc-port and --mc-mode=headless are mutually exclusive." >&2
        echo "       Edit installation/fabric_server_config.json to change the headless port," >&2
        echo "       or drop --mc-mode=headless to use an explicit port." >&2
        exit 1
    fi

    HEADLESS_SERVER_STARTED=false
    if [[ "$MC_MODE" == "headless" ]]; then
        echo "[mc] mode=headless — setting up + starting managed Fabric server..."

        # One-time setup (idempotent): installs the Fabric server jar if absent.
        if [[ ! -f "installation/fabric_server/fabric-server-launcher.jar" ]]; then
            echo "[mc] First-time setup; downloading Fabric server (this takes a minute)..."
            bash installation/setup_fabric_server.sh || {
                echo "ERROR: Fabric server setup failed." >&2
                exit 1
            }
        fi

        # Start the server in daemon mode; --daemon prints the port on stdout.
        HEADLESS_MC_PORT=$(bash installation/run_fabric_server.sh --daemon) || {
            echo "ERROR: Failed to start headless Fabric server." >&2
            exit 1
        }
        HEADLESS_SERVER_STARTED=true
        MC_PORT_OVERRIDE="$HEADLESS_MC_PORT"
        echo "[mc] Headless server ready on port $HEADLESS_MC_PORT"

        # Make sure we shut the server down on script exit / Ctrl-C.
        trap 'if [[ "$HEADLESS_SERVER_STARTED" == "true" ]]; then echo "[mc] Stopping headless Fabric server..."; bash installation/run_fabric_server.sh --stop || true; fi' EXIT INT TERM
    fi

    # Build arguments for Python
    PYTHON_ARGS=""

    if [[ -n "$PLANNER_MODE" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --planner-mode=$PLANNER_MODE"
    fi

    # Feature toggles
    if [[ "$NO_OPTIMIZER" == "true" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --no-optimizer"
    fi
    if [[ "$NO_REFACTOR" == "true" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --no-refactor"
    fi
    if [[ "$POSTMILESTONE_ADAPTIVE" == "true" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --postmilestone-adaptive"
    fi
    if [[ "$INCLUDE_SKILL_CODE" == "true" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --include-skill-code"
    fi
    if [[ "$PURE_REASONING" == "true" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --pure-reasoning"
    fi
    if [[ "$COMBAT" == "true" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --combat"
    fi

    # Run options
    if [[ "$CURRICULUM_AGENT_MODE" == "manual" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --manual"
    fi
    if [[ "$RUN_MODE" == "inference" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --inference"
    fi
    if [[ -n "$CKPT_DIR" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --ckpt-dir=$CKPT_DIR"
    fi
    if [[ -n "$MODEL_BACKEND" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --model=$MODEL_BACKEND"
    fi
    # Experiment parameters
    if [[ -n "$OPT_THRESHOLD" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --optimization-threshold=$OPT_THRESHOLD"
    fi
    if [[ -n "$OPT_EPSILON" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --optimization-epsilon=$OPT_EPSILON"
    fi
    if [[ -n "$OPT_GAMMA" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --optimization-gamma=$OPT_GAMMA"
    fi
    if [[ -n "$MAX_SKILLS" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --max-skills=$MAX_SKILLS"
    fi
    if [[ -n "$GRAPH_SNAPSHOT_INTERVAL" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --graph-snapshot-interval=$GRAPH_SNAPSHOT_INTERVAL"
    fi
    if [[ -n "$MAX_ITERATIONS" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --max-iterations=$MAX_ITERATIONS"
    fi
    if [[ -n "$RESUME_FLAG" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS $RESUME_FLAG"
    fi
    if [[ -n "$DEBUG_FLAG" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS $DEBUG_FLAG"
    fi
    if [[ -n "$MC_PORT_OVERRIDE" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --mc-port=$MC_PORT_OVERRIDE"
    fi
    if [[ -n "$DOMAIN" ]]; then
        PYTHON_ARGS="$PYTHON_ARGS --domain=$DOMAIN"
    fi

    # Show selected modes
    echo "Curriculum agent mode: ${CURRICULUM_AGENT_MODE:-auto}"
    echo "Run mode: ${RUN_MODE:-learn}"
    if [[ -n "$CKPT_DIR" ]]; then
        echo "Checkpoint dir: $CKPT_DIR"
    fi

    # Run the application
    echo "Starting SkillNet..."
    python -m skillnet $PYTHON_ARGS || {
        echo "Error: Failed to start application"
        exit 1
    }
    exit 0
}

# Function to install the application
install_app() {
    echo "Starting installation..."

    # Use the current Python (should be from conda if activated)
    PYTHON_CMD="python"
    if ! command -v python &> /dev/null; then
        PYTHON_CMD="python3"
        if ! command -v python3 &> /dev/null; then
            echo "Error: Python is not found. Please activate your conda environment or install Python."
            exit 1
        fi
    fi

    # Get the exact version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo "Checking Python version..."
    echo "Found Python version $PYTHON_VERSION"
    
    # Warn if not Python 3.10 (but don't fail)
    PYTHON_MAJOR_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f1,2)
    if [[ "$PYTHON_MAJOR_MINOR" != "3.10" ]]; then
        echo "Warning: Python 3.10.x is recommended, but using $PYTHON_VERSION"
    fi

    # Run configuration script
    echo ""
    $PYTHON_CMD config.py || {
        echo "Error: Configuration failed. Please try again."
        exit 1
    }
    echo ""

    # Check for Node.js
    echo "Checking Node.js version..."
    if ! command -v node &> /dev/null; then
        echo "Node.js is not installed. Please install Node.js."
        exit 1
    fi

    NODE_VERSION=$(node -v | sed 's/v//')
    echo "Found Node.js version $NODE_VERSION"
    
    # Parse the major version number
    NODE_MAJOR_VERSION=$(echo "$NODE_VERSION" | cut -d. -f1)
    if [[ "$NODE_MAJOR_VERSION" -lt 10 ]]; then
        echo "Node.js version is lower than 10. Please install Node.js 10 or later."
        exit 1
    fi

    # Check if conda environment is active
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        echo ""
        echo "Warning: No conda environment detected. Please activate your conda environment first."
        echo "Example: conda activate <your_env_name>"
        echo ""
        echo "Continuing with current Python environment..."
    else
        echo "Using conda environment: $CONDA_DEFAULT_ENV"
    fi

    # Install dependencies in editable mode
    echo "Installing Python dependencies..."
    if [[ -n "$VERBOSE" ]]; then
        pip install -e . || {
            echo "Error: Failed to install Python dependencies"
            exit 1
        }
    else
        pip install -e . >/dev/null 2>&1 || {
            echo "Error: Failed to install Python dependencies"
            exit 1
        }
    fi

    # Install root Node.js dependencies (@babel/* for skill code analysis)
    echo "Installing root Node.js dependencies..."
    if [[ -n "$VERBOSE" ]]; then
        npm install || {
            echo "Error: Failed to install root Node.js dependencies"
            exit 1
        }
    else
        npm install $NPM_QUIET >/dev/null 2>&1 || {
            echo "Error: Failed to install root Node.js dependencies"
            exit 1
        }
    fi

    # Change to the mineflayer environment directory
    echo "Setting up Mineflayer environment..."
    if [[ ! -d "skillnet/domains/minecraft/action_space/env/mineflayer" ]]; then
        echo "Error: Could not find mineflayer directory"
        exit 1
    fi
    pushd skillnet/domains/minecraft/action_space/env/mineflayer >/dev/null || {
        echo "Error: Could not change to mineflayer directory"
        exit 1
    }

    # Check if npx is available (it comes with Node.js)
    if ! command -v npx &> /dev/null; then
        echo "Installing npx..."
        if [[ -n "$VERBOSE" ]]; then
            npm install -g npx || {
                echo "Error: Failed to install npx"
                exit 1
            }
        else
            npm install -g npx $NPM_QUIET >/dev/null 2>&1 || {
                echo "Error: Failed to install npx"
                exit 1
            }
        fi
    else
        echo "npx is already available"
    fi

    # Install Node.js dependencies
    echo "Installing Node.js dependencies..."
    if [[ -n "$VERBOSE" ]]; then
        npm install || {
            echo "Error: Failed to install Node.js dependencies"
            exit 1
        }
    else
        npm install $NPM_QUIET >/dev/null 2>&1 || {
            echo "Error: Failed to install Node.js dependencies"
            exit 1
        }
    fi

    # Change to the mineflayer-collectblock directory and compile
    echo "Setting up mineflayer-collectblock..."
    if [[ ! -d "mineflayer-collectblock" ]]; then
        echo "Error: Could not find mineflayer-collectblock directory"
        exit 1
    fi
    cd mineflayer-collectblock || {
        echo "Error: Could not change to mineflayer-collectblock directory"
        exit 1
    }
    
    # Install dependencies in mineflayer-collectblock directory first
    echo "Installing mineflayer-collectblock dependencies..."
    if [[ -n "$VERBOSE" ]]; then
        npm install || {
            echo "Error: Failed to install mineflayer-collectblock dependencies"
            exit 1
        }
    else
        npm install $NPM_QUIET >/dev/null 2>&1 || {
            echo "Error: Failed to install mineflayer-collectblock dependencies"
            exit 1
        }
    fi
    
    # Compile TypeScript
    if [[ -n "$VERBOSE" ]]; then
        npx tsc || {
            echo "Error: TypeScript compilation failed"
            exit 1
        }
    else
        npx tsc >/dev/null 2>&1 || {
            echo "Error: TypeScript compilation failed. Run with --verbose to see detailed errors."
            exit 1
        }
    fi

    # Return to the mineflayer directory
    cd ..
    if [[ -n "$VERBOSE" ]]; then
        npm install || {
            echo "Error: Failed to install remaining Node.js dependencies"
            exit 1
        }
    else
        npm install $NPM_QUIET >/dev/null 2>&1 || {
            echo "Error: Failed to install remaining Node.js dependencies"
            exit 1
        }
    fi

    # Return to the previous directory
    popd >/dev/null

    # Re-apply hand-patches to node_modules (mineflayer-pathfinder + mineflayer/place_block).
    # See installation/patches/ + installation/apply_patches.sh for details.
    if [[ -x "installation/apply_patches.sh" ]]; then
        echo ""
        echo "Applying node_modules patches..."
        installation/apply_patches.sh || {
            echo "Error: Failed to apply node_modules patches"
            exit 1
        }
    fi

    echo ""
    echo "Installation completed successfully!"
    echo "Run 'skillnet.sh' without parameters to start the application."
    echo ""
    exit 0
}

# Parse command line arguments
PLANNER_MODE=""
NO_OPTIMIZER=""
NO_REFACTOR=""
POSTMILESTONE_ADAPTIVE=""
INCLUDE_SKILL_CODE=""
COMBAT=""
CURRICULUM_AGENT_MODE="auto"
RUN_MODE="learn"
CKPT_DIR=""
DOMAIN=""
DO_INSTALL=false

for arg in "$@"; do
    case "$arg" in
        install)
            DO_INSTALL=true
            ;;
        --planner-mode=*)
            PLANNER_MODE="${arg#*=}"
            ;;
        # Feature toggles
        --no-optimizer)
            NO_OPTIMIZER="true"
            ;;
        --no-refactor)
            NO_REFACTOR="true"
            ;;
        --postmilestone-adaptive)
            POSTMILESTONE_ADAPTIVE="true"
            ;;
        --include-skill-code)
            INCLUDE_SKILL_CODE="true"
            ;;
        --pure-reasoning)
            PURE_REASONING="true"
            ;;
        --combat)
            COMBAT="true"
            ;;
        # Experiment parameters
        --optimization-threshold=*)
            OPT_THRESHOLD="${arg#*=}"
            ;;
        --optimization-epsilon=*)
            OPT_EPSILON="${arg#*=}"
            ;;
        --optimization-gamma=*)
            OPT_GAMMA="${arg#*=}"
            ;;
        --max-skills=*)
            MAX_SKILLS="${arg#*=}"
            ;;
        --graph-snapshot-interval=*)
            GRAPH_SNAPSHOT_INTERVAL="${arg#*=}"
            ;;
        --max-iterations=*)
            MAX_ITERATIONS="${arg#*=}"
            ;;
        --resume)
            RESUME_FLAG="--resume"
            ;;
        --no-resume)
            RESUME_FLAG="--no-resume"
            ;;
        --debug)
            DEBUG_FLAG="--debug"
            ;;
        --no-debug)
            DEBUG_FLAG="--no-debug"
            ;;
        # Run options
        --manual)
            CURRICULUM_AGENT_MODE="manual"
            ;;
        --auto)
            CURRICULUM_AGENT_MODE="auto"
            ;;
        --inference)
            RUN_MODE="inference"
            ;;
        --learn)
            RUN_MODE="learn"
            ;;
        --ckpt-dir=*)
            CKPT_DIR="${arg#*=}"
            ;;
        --model=*)
            MODEL_BACKEND="${arg#*=}"
            ;;
        --verbose)
            VERBOSE="1"
            ;;
        --mc-port=*)
            MC_PORT_OVERRIDE="${arg#*=}"
            ;;
        --mc-mode=*)
            MC_MODE="${arg#*=}"
            ;;
        --domain=*)
            DOMAIN="${arg#*=}"
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            echo "Unknown argument: $arg"
            show_usage
            ;;
    esac
done

# Set NPM_QUIET based on VERBOSE flag
if [[ -n "$VERBOSE" ]]; then
    NPM_QUIET=""
fi

if [[ "$DO_INSTALL" == true ]]; then
    install_app
else
    run_app
fi
