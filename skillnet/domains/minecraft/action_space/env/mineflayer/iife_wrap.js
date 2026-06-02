/**
 * IIFE wrap for concatenated skill programs.
 *
 * Why: PSN concatenates many skill files into one `eval(fullCode)` string. If
 *      any skill writes a module-top-level `let mcData = null;` (which Phase 2
 *      LLM has been observed to do — see v12 line 5286), it collides with
 *      globalDepsCode's `var mcData = global.mcData;` → SyntaxError "Identifier
 *      'mcData' has already been declared". Two skills both with module-top
 *      `let X` also collide with each other.
 *
 * Fix:  Wrap each top-level FunctionDeclaration in its own IIFE, and pull any
 *       preceding non-function top-level statements (variable decls, comments)
 *       into THAT function's IIFE. Module-top `let mcData = null;` from skill
 *       N gets bundled with skill N's function declaration. We export each
 *       function to `globalThis` so cross-skill bare-name calls keep working
 *       (lookup walks scope chain → globalThis).
 *
 * Why acorn tokenizer (not parser):
 *   - The full parser rejects duplicate `let` declarations (semantic error),
 *     which is the very condition we're trying to fix. We need a non-validating
 *     scan that gives us function-boundary positions.
 *   - esprima 4 used in earlier attempts doesn't support optional chaining
 *     (`?.`), which appears in real v12 skill code (e.g. setupCraftingTable.js).
 *   - acorn 7.4 supports ES2022 syntax AND has a tokenizer() API that produces
 *     tokens without semantic validation.
 *
 * Algorithm (token-based):
 *   1. Tokenize entire programs string with acorn.tokenizer (ES2022).
 *   2. Walk tokens. Whenever we hit a `function` keyword AT TOP LEVEL
 *      (paren-depth 0 AND brace-depth 0), the function's name is the next
 *      `name` token. Then walk forward to find body `{` (first `{` AFTER
 *      balanced params `()`). From there, brace-balance walk to find matching
 *      `}` (counting `${` from template interp also as +1, `{` as +1, `}` as -1,
 *      ignoring all brace/paren changes while paren-depth > 0).
 *   3. Chunk boundaries: chunk_i = [end_of_chunk_(i-1) .. end_of_fn_i_body].
 *      First chunk starts at programs[0] (covers any leading content like
 *      module-top `let mcData = null;` before the first function).
 *   4. Trailing content after last function appends to the last chunk.
 *
 * Wrap each chunk as:
 *   (function() {
 *     <chunk_text>
 *     if (typeof NAME === 'function') globalThis.NAME = NAME;
 *   })();
 *
 * Defensive fallbacks (return programs unchanged):
 *   - Empty / non-string input.
 *   - Tokenizer throws (e.g. truly invalid syntax — let downstream eval surface).
 *   - No top-level FunctionDeclaration found.
 */

const acorn = require('acorn');

/**
 * @param {string} programs Concatenated skill code + control primitives.
 * @returns {string} Same programs with each top-level function in an IIFE.
 */
function wrapProgramsInIIFE(programs) {
    if (typeof programs !== 'string' || programs.length === 0) {
        return programs;
    }

    // ----------------------------------------------------------------
    // Step 1: Tokenize. Use ES2022 to support optional chaining (?.).
    // ----------------------------------------------------------------
    const tokens = [];
    try {
        for (const t of acorn.tokenizer(programs, { ecmaVersion: 2022 })) {
            tokens.push({
                type: t.type.label,
                value: t.value,
                start: t.start,
                end: t.end,
            });
        }
    } catch (e) {
        console.error('[iife_wrap] tokenize failed; returning programs unchanged:', e.message);
        return programs;
    }

    // ----------------------------------------------------------------
    // Step 2: Walk tokens, identify each TOP-LEVEL function declaration
    // (function keyword at paren-depth 0 AND brace-depth 0). For each,
    // record [startTokenIdx, bodyEndPos] including the closing `}`.
    // ----------------------------------------------------------------
    const fns = [];  // [{ name, end }] — `end` is source position AFTER closing `}`
    let parenDepth = 0;
    let braceDepth = 0;

    for (let i = 0; i < tokens.length; i++) {
        const t = tokens[i];

        // Track top-level paren/brace depth so we recognize "top-level"
        // function declarations (vs nested inside another function body
        // or inside parens like callback args).
        if (t.type === '(') { parenDepth++; continue; }
        if (t.type === ')') { parenDepth--; continue; }
        if (parenDepth === 0) {
            if (t.type === '{') { braceDepth++; continue; }
            if (t.type === '${') { braceDepth++; continue; }
            if (t.type === '}') { braceDepth--; continue; }
        }

        if (t.type !== 'function') continue;
        if (parenDepth !== 0 || braceDepth !== 0) continue;

        // This is a top-level function keyword. The next token should be
        // an identifier (the function name).
        const nameTok = tokens[i + 1];
        if (!nameTok || nameTok.type !== 'name') {
            // Anonymous function (function expression) — not a declaration
            continue;
        }
        const fnName = nameTok.value;

        // From here, find the body `{` (first `{` after balanced params `()`).
        // Then brace-balance walk to find the matching `}`.
        let j = i + 2;
        let pd = 0;
        let bd = 0;
        let foundBody = false;
        let bodyEndPos = -1;
        while (j < tokens.length) {
            const u = tokens[j];
            if (u.type === '(') pd++;
            else if (u.type === ')') pd--;
            else if (pd === 0) {
                if (u.type === '{') {
                    foundBody = true;
                    bd++;
                } else if (u.type === '${') {
                    if (foundBody) bd++;
                } else if (u.type === '}') {
                    if (foundBody) {
                        bd--;
                        if (bd === 0) {
                            bodyEndPos = u.end;
                            break;
                        }
                    }
                }
            }
            j++;
        }

        if (bodyEndPos === -1) {
            // Unbalanced — bail to defensive fallback. The eval downstream
            // will surface the real syntax error.
            console.error('[iife_wrap] unbalanced braces for fn', fnName, '; returning programs unchanged');
            return programs;
        }

        fns.push({ name: fnName, end: bodyEndPos });
        // Advance `i` PAST the function body so we don't accidentally re-detect
        // nested function declarations as top-level. We're tracking top-level
        // brace depth in the outer loop, but the moment we increment braceDepth
        // upon entering this fn's body, we'd already skip the body's contents
        // via parenDepth+braceDepth gates — so the next top-level fn detection
        // only fires after we exit this body. But to be safe AND efficient,
        // jump i to j (the closing-brace token of this fn). The outer loop's
        // braceDepth tracking will see the close as a normal `}` and decrement.
        // Actually the outer loop only updates braceDepth via the `if (parenDepth===0)` branch
        // BEFORE the `function` check; here we already consumed `i`. Let i advance naturally —
        // but we need to make sure outer loop sees the `{` and `}` of THIS body as braceDepth changes.
        // The outer loop processes them via the early `continue` paths above.
    }

    // ----------------------------------------------------------------
    // Step 3: Build chunks based on detected fn boundaries.
    // ----------------------------------------------------------------
    if (fns.length === 0) {
        return programs;
    }

    const chunks = [];
    let chunkStart = 0;
    for (const fn of fns) {
        chunks.push({
            text: programs.slice(chunkStart, fn.end),
            name: fn.name,
        });
        chunkStart = fn.end;
    }
    // Trailing content (rare; usually blank lines) appended to last chunk.
    if (chunkStart < programs.length) {
        chunks[chunks.length - 1].text += programs.slice(chunkStart);
    }

    // ----------------------------------------------------------------
    // Step 4: Emit per-chunk IIFE with explicit globalThis export.
    // ----------------------------------------------------------------
    return chunks.map(({ text, name }) => {
        const safeName = name.replace(/[^\w$]/g, '_');
        return [
            '(function() {',
            text,
            `if (typeof ${safeName} === 'function') globalThis.${safeName} = ${safeName};`,
            '})();',
        ].join('\n');
    }).join('\n\n');
}

module.exports = { wrapProgramsInIIFE };
