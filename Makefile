.PHONY: install lint run smoke verify clean

install:
	pip install -e .
	npm install
	cd skillnet/domains/minecraft/action_space/env/mineflayer && npm install
	# Re-apply hand-patches to node_modules (mineflayer-pathfinder corner-pinch
	# escape + mineflayer place_block failure-diagnostics). These are required
	# for PSN to navigate / report placement failures correctly; without them
	# pathfinder times out ~50% of obstacle-corridor moves and placeBlock
	# failures surface as opaque timeouts. See installation/patches/.
	bash installation/apply_patches.sh

lint:
	python -m py_compile $$(find skillnet -name '*.py' -not -path '*/node_modules/*' -not -path '*/__pycache__/*')

run:
	./skillnet.sh

smoke:
	PYTHONPATH=. python -c "import skillnet; from skillnet.core.config import PSNConfig; print('PSN import graph OK')"
	PYTHONPATH=. python -c "import skillnet.psn; import sys; \
assert 'skillnet.domains.minecraft.action_space.env.bridge' not in sys.modules, \
'PSN core re-imported a Minecraft-specific module — keep skillnet.psn domain-agnostic'; \
print('PSN core import-isolation OK')"
	@bad=$$(grep -rln --exclude-dir=node_modules --exclude-dir=__pycache__ 'from javascript import' skillnet --include='*.py' 2>/dev/null | grep -v 'skillnet/languages/javascript\.py' || true); \
test -z "$$bad" && echo "No javascript-bridge leak outside skillnet/languages/javascript.py" || (echo "ERROR: javascript-bridge import leaked into:"; echo "$$bad"; exit 1)

verify:
	@echo "==> [1/5] Python import graph"
	@$(MAKE) -s smoke
	@echo "==> [2/5] Mineflayer Node deps"
	@test -d skillnet/domains/minecraft/action_space/env/mineflayer/node_modules/mineflayer && echo "mineflayer node_modules present" || (echo "ERROR: run 'make install' first — mineflayer node_modules missing"; exit 1)
	@echo "==> [3/5] Mineflayer patches applied"
	@grep -q "PATCH (skillnet" skillnet/domains/minecraft/action_space/env/mineflayer/node_modules/mineflayer-pathfinder/index.js 2>/dev/null && echo "pathfinder patch applied" || (echo "ERROR: pathfinder patch missing — run 'bash installation/apply_patches.sh'"; exit 1)
	@echo "==> [4/5] Credentials in .env"
	@test -f .env && grep -Eq '^(OPENAI_API_KEY|VLLM_API_BASE)=.+' .env && echo ".env has OPENAI_API_KEY or VLLM_API_BASE" || (echo "ERROR: .env missing or has no OPENAI_API_KEY / VLLM_API_BASE — copy from .env.example"; exit 1)
	@echo "==> [5/5] Java runtime (only needed for headless Fabric server path)"
	@command -v java >/dev/null 2>&1 && java -version 2>&1 | head -1 || echo "NOTE: java not found — only required if you'll run a local Fabric server; Modrinth/desktop path does not need it"
	@echo "==> verify OK"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
