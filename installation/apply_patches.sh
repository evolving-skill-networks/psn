#!/bin/bash
# Re-apply all skillnet node_modules patches after a fresh `npm install`.
#
# Each .patch in installation/patches/ is a standard unified diff against the
# pristine upstream version recorded in the patch header. `patch --dry-run`
# is used to check idempotency: if a patch is already applied, skip it.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$REPO_ROOT/installation/patches"
NM_ROOT="$REPO_ROOT/skillnet/domains/minecraft/action_space/env/mineflayer/node_modules"

if [ ! -d "$NM_ROOT" ]; then
    echo "ERROR: node_modules not found at $NM_ROOT"
    echo "       run 'npm install' in skillnet/domains/minecraft/action_space/env/mineflayer/ first"
    exit 1
fi

# Map each .patch to the file it applies to. Keeping this explicit (vs. parsing
# the diff header) makes it obvious what we're touching.
declare -A TARGETS=(
    [mineflayer-pathfinder-index.js.patch]="mineflayer-pathfinder/index.js"
    [mineflayer-place_block.js.patch]="mineflayer/lib/plugins/place_block.js"
    [mineflayer-entities.js.patch]="mineflayer/lib/plugins/entities.js"
)

applied=0
skipped=0
failed=0

for patch_name in "${!TARGETS[@]}"; do
    patch_file="$PATCH_DIR/$patch_name"
    rel_target="${TARGETS[$patch_name]}"
    target="$NM_ROOT/$rel_target"

    if [ ! -f "$patch_file" ]; then
        echo "MISSING patch file: $patch_file"; failed=$((failed+1)); continue
    fi
    if [ ! -f "$target" ]; then
        echo "MISSING target file: $target"; failed=$((failed+1)); continue
    fi

    # Dry-run forward apply: does the target currently match the pre-patch state?
    if patch -p1 --dry-run --forward --silent -d "$NM_ROOT" < "$patch_file" > /dev/null 2>&1; then
        patch -p1 --forward --silent -d "$NM_ROOT" < "$patch_file"
        echo "APPLIED  $rel_target"
        applied=$((applied+1))
        continue
    fi

    # Dry-run reverse apply: is the patch already applied?
    if patch -p1 --dry-run --reverse --silent -d "$NM_ROOT" < "$patch_file" > /dev/null 2>&1; then
        echo "SKIP     $rel_target (already patched)"
        skipped=$((skipped+1))
        continue
    fi

    # Neither direction worked -> target has drifted from both pristine and
    # patched states, probably because upstream shipped a new version.
    echo "FAILED   $rel_target"
    echo "         neither pristine nor already-patched match — upstream drift?"
    echo "         update $patch_name manually"
    failed=$((failed+1))
done

echo ""
echo "summary: $applied applied, $skipped already patched, $failed failed"
[ "$failed" -eq 0 ]
