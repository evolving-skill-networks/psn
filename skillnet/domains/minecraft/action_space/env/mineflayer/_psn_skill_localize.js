'use strict';
/*
 * Skill-level localization for runtime errors.
 *
 * The env evals the concatenated `programs` blob as `(async () => {<programs>
 * \n<code>})()`. When a skill throws, handleError historically reported only the
 * blob LINE ("programs line N"), never WHICH SKILL the line belongs to -- so the
 * optimizer's credit-assignment had to guess the culprit from the call chain and
 * frequently misattributed (e.g. "fixing" a correct sibling counter while the
 * real buggy skill went untouched -> non-convergence).
 *
 * The information is already in the V8 stack: the throwing skill is a NAMED
 * function, e.g.
 *   at countItems (eval at evaluateCode (.../index.js:NNN), <anonymous>:7:32)
 * Anonymous callbacks (an Array.reduce arrow, etc.) show as `at eval (...)` /
 * `at <anonymous> (...)` / `at Array.reduce (<anonymous>)` and must be skipped:
 * we want the nearest NAMED skill frame. `evaluateCode` knows `skillNames`, so we
 * prefer an exact match against the registered skills.
 */

// V8 eval frame: "    at <fn> (eval at <outer> (<file>:L:C), <anonymous>:L:C)".
// `.*` spans the inner "(<file>:L:C)" before the trailing "<anonymous>:L:C)".
const EVAL_FRAME_RE = /^\s*at (?:async )?([\w$.]+) \(eval .*<anonymous>:\d+:\d+\)/;

// Reject anonymous/builtin frame names when no skillNames list is supplied.
function isNamedSkill(fn) {
  if (!fn) return false;
  if (fn === 'eval' || fn === '<anonymous>') return false;
  if (fn.startsWith('Array.') || fn.startsWith('Promise.') ||
      fn.startsWith('Function.') || fn.startsWith('Object.<')) return false;
  return /^[A-Za-z_$][\w$]*$/.test(fn);
}

/**
 * Recover the skill (named function) at which a runtime error was thrown.
 * @param {string} stack    err.stack from the eval'd program
 * @param {string[]} [skillNames] registered skill names to match exactly
 * @returns {string|null} the buggy skill name, or null if not localizable
 */
function localizeSkillFromStack(stack, skillNames) {
  if (!stack || typeof stack !== 'string') return null;
  const nameSet = new Set(Array.isArray(skillNames) ? skillNames : []);
  for (const line of stack.split('\n')) {
    const m = EVAL_FRAME_RE.exec(line);
    if (!m) continue;
    // V8 prefixes methods with their holder, e.g. "Object.ensureCobble".
    const fn = m[1].replace(/^Object\./, '');
    if (nameSet.size ? nameSet.has(fn) : isNamedSkill(fn)) return fn;
  }
  return null;
}

/**
 * Recover the full named-frame chain at which a runtime error was thrown,
 * distinguishing the deepest INLINE culprit (the actual throwing function, even
 * when it is a helper that is not a registered skill) from the nearest
 * REGISTERED ancestor (a real skill that feedback can be routed to). When the
 * culprit is itself a registered skill the two coincide.
 *
 * @param {string} stack    err.stack from the eval'd program
 * @param {string[]} [skillNames] registered skill names
 * @returns {{culprit: string, registeredAncestor: (string|null), chain: string[]}|null}
 *   chain is ordered deepest (innermost) frame first; null if not localizable.
 */
function localizeSkillChain(stack, skillNames) {
  if (!stack || typeof stack !== 'string') return null;
  const nameSet = new Set(Array.isArray(skillNames) ? skillNames : []);
  const chain = [];
  for (const line of stack.split('\n')) {
    const m = EVAL_FRAME_RE.exec(line);
    if (!m) continue;
    const fn = m[1].replace(/^Object\./, '');
    if (isNamedSkill(fn)) chain.push(fn);
  }
  if (!chain.length) return null;
  const registeredAncestor = chain.find((fn) => nameSet.has(fn)) || null;
  return { culprit: chain[0], registeredAncestor, chain };
}

module.exports = { localizeSkillFromStack, localizeSkillChain, isNamedSkill };
