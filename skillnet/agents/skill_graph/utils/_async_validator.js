/**
 * Babel-based static detector for concurrency-risky async-call patterns.
 *
 * Invocation: node _async_validator.js <code_file.js>
 * Output: JSON on stdout.
 *
 * Detects async function calls that execute fire-and-forget inside a
 * synchronous callback passed as an argument. Such patterns cause
 * unbounded concurrent async work (e.g. concurrent pathfinder goals)
 * which starves mineflayer's event loop and crashes the process.
 *
 * NOTE: We flag calls inside sync callbacks regardless of .then/.catch/.finally
 * because the concurrency risk is orthogonal to rejection handling.
 */

const parser = require("@babel/parser");
const traverse = require("@babel/traverse").default;

// ──────────────────────────────────────────────────────────────────────
// Known-async primitives/methods in the Minecraft domain.
// User-declared `async function X(...)` is discovered by walking the AST.
// ──────────────────────────────────────────────────────────────────────
const KNOWN_ASYNC_FUNCS = new Set([
  "mineBlock", "exploreUntil", "craftItem", "placeItem", "killMob",
  "useChest", "smeltItem", "shoot", "gotoWithTimeout", "collectWithTimeout",
  "givePlacedItemBack", "waitForMobRemoved",
]);
const ASYNC_BOT_MEMBERS = new Set([
  "dig", "equip", "placeBlock", "consume", "activateBlock",
  "activateItem", "fish", "sleep", "useOn", "craft", "waitForTicks",
]);
const ASYNC_BOT_PATHFINDER = new Set(["goto"]);

function parse(code) {
  try {
    return parser.parse(code, {
      sourceType: "module",
      plugins: [],
      errorRecovery: true,
    });
  } catch (e) {
    return { __parseError: e.message };
  }
}

function discoverAsyncDecls(ast) {
  const decls = new Set();
  traverse(ast, {
    FunctionDeclaration(path) {
      if (path.node.async && path.node.id) decls.add(path.node.id.name);
    },
    FunctionExpression(path) {
      if (path.node.async && path.node.id) decls.add(path.node.id.name);
    },
    VariableDeclarator(path) {
      const init = path.node.init;
      if (init && (init.type === "ArrowFunctionExpression" || init.type === "FunctionExpression") && init.async) {
        if (path.node.id && path.node.id.name) decls.add(path.node.id.name);
      }
    },
  });
  return decls;
}

// Scope-aware async-binding check.
// Returns true iff the NEAREST lexical declaration of `name` (looked up via
// Babel's path.scope.getBinding) is an async function/arrow/expression.
// This replaces the previous flat-Set lookup, which treated any call by
// a name appearing as `async function NAME` ANYWHERE in the source as
// async — false-positive whenever an inner sync helper shadowed an outer
// async same-name (e.g. HelperExtractor leaves the local sync def in
// ensureWoodLogs while also registering an extracted async findTreeBlock).
function isAsyncByBinding(callPath, name) {
  const binding = callPath.scope && callPath.scope.getBinding(name);
  if (!binding) return false;
  const decl = binding.path && binding.path.node;
  if (!decl) return false;
  if (decl.type === "FunctionDeclaration") return !!decl.async;
  if (decl.type === "VariableDeclarator") {
    const init = decl.init;
    if (init && (init.type === "ArrowFunctionExpression" || init.type === "FunctionExpression")) {
      return !!init.async;
    }
  }
  if (decl.type === "FunctionExpression" || decl.type === "ArrowFunctionExpression") {
    return !!decl.async;
  }
  return false;
}

function calleeToName(callee, callPath) {
  if (callee.type === "Identifier") {
    if (KNOWN_ASYNC_FUNCS.has(callee.name)) return callee.name;
    if (isAsyncByBinding(callPath, callee.name)) return callee.name;
    return null;
  }
  if (callee.type === "MemberExpression") {
    if (callee.object && callee.object.name === "bot" &&
        callee.property && ASYNC_BOT_MEMBERS.has(callee.property.name)) {
      return "bot." + callee.property.name;
    }
    if (callee.object && callee.object.type === "MemberExpression" &&
        callee.object.object && callee.object.object.name === "bot" &&
        callee.object.property && callee.object.property.name === "pathfinder" &&
        callee.property && ASYNC_BOT_PATHFINDER.has(callee.property.name)) {
      return "bot.pathfinder." + callee.property.name;
    }
  }
  return null;
}

// Methods whose callback RETURN VALUE escapes to the caller: value-collecting
// iterations (map/flatMap collect into an array, usually for Promise.all) and
// promise-chain methods (then/catch/finally chain the returned promise). A call
// returned out of such a callback is NOT fire-and-forget — its promise is handed
// to the consumer, which can await it. (forEach/filter/reduce are deliberately
// excluded: forEach discards the value, filter wants a boolean, reduce an acc.)
const RETURN_ESCAPING_METHODS = new Set(["map", "flatMap", "then", "catch", "finally"]);

// If `fnPath` (an arrow/function expression) is the callback ARGUMENT of a
// CallExpression whose method is return-escaping, return that host CallExpression
// path so the climb can continue from it (e.g. await Promise.all(arr.map(fn))).
function returnEscapingHostCall(fnPath) {
  if (!fnPath) return null;
  const parent = fnPath.parentPath;
  if (parent && parent.node.type === "CallExpression" &&
      parent.node.arguments.includes(fnPath.node) &&
      parent.node.callee.type === "MemberExpression" &&
      parent.node.callee.property &&
      RETURN_ESCAPING_METHODS.has(parent.node.callee.property.name)) {
    return parent;
  }
  return null;
}

// Climb up from a CallExpression node; determine if the whole expression
// is eventually awaited (either directly, after a .then/.catch/.finally chain,
// inside a Promise.all call that is awaited, or RETURNED out of a value-collecting
// callback whose host iteration is itself awaited).
function isAwaited(callPath) {
  let p = callPath;
  while (p && p.parentPath) {
    const parent = p.parentPath.node;
    const parentPath = p.parentPath;
    if (parent.type === "AwaitExpression") return true;
    // Chain: X().then(...) — keep going up
    if (parent.type === "MemberExpression" && parent.object === p.node) {
      p = p.parentPath;
      continue;
    }
    if (parent.type === "CallExpression") {
      const callee = parent.callee;
      // Promise.all([..., X(), ...]).then(...) / await Promise.all(...) — keep going
      if (callee && callee.type === "MemberExpression" &&
          callee.object && callee.object.name === "Promise" &&
          callee.property && ["all", "race", "any", "allSettled"].includes(callee.property.name)) {
        p = p.parentPath;
        continue;
      }
      // We are the callee of a chain method call (e.g. X().then(cb) — we are X().then)
      if (parent.callee === p.node) {
        p = p.parentPath;
        continue;
      }
    }
    // The promise is RETURNED out of a callback: `(x) => CALL` (expression body)
    // or `(x) => { return CALL; }` (block body). If that callback is passed to a
    // return-escaping method (map/flatMap/then/...), continue from the host call
    // so an outer `await Promise.all(arr.map(fn))` / awaited chain clears it.
    let returnedFnPath = null;
    if (parent.type === "ArrowFunctionExpression" && parent.body === p.node) {
      returnedFnPath = parentPath;
    } else if (parent.type === "ReturnStatement" && parent.argument === p.node) {
      returnedFnPath = enclosingFunctionPath(parentPath);
    }
    if (returnedFnPath) {
      const host = returnEscapingHostCall(returnedFnPath);
      if (host) { p = host; continue; }
    }
    break;
  }
  return false;
}

// Climb to the nearest enclosing arrow/function expression PATH (not declaration
// boundary semantics — just the lexical container), starting from `path`.
function enclosingFunctionPath(path) {
  let p = path && path.parentPath;
  while (p) {
    const t = p.node.type;
    if (t === "ArrowFunctionExpression" || t === "FunctionExpression" || t === "FunctionDeclaration") {
      return p;
    }
    p = p.parentPath;
  }
  return null;
}

// Does the expression have .then/.catch/.finally anywhere in its chain?
function hasThenCatch(callPath) {
  let p = callPath;
  while (p && p.parentPath) {
    const parent = p.parentPath.node;
    if (parent.type === "MemberExpression" && parent.object === p.node) {
      if (parent.property && ["then", "catch", "finally"].includes(parent.property.name)) return true;
      p = p.parentPath;
      continue;
    }
    if (parent.type === "CallExpression" && parent.callee === p.node) {
      p = p.parentPath;
      continue;
    }
    break;
  }
  return false;
}

function enclosingFunction(path) {
  let p = path.parentPath;
  while (p) {
    const t = p.node.type;
    if (t === "ArrowFunctionExpression" || t === "FunctionExpression" || t === "FunctionDeclaration") {
      return { path: p, async: !!p.node.async };
    }
    p = p.parentPath;
  }
  return null;
}

// A sync callback = non-async arrow/fn expression that is passed as an argument.
function isSyncCallback(fnInfo) {
  if (!fnInfo || fnInfo.async) return false;
  const parent = fnInfo.path.parentPath;
  if (!parent) return false;
  if (parent.node.type === "CallExpression" && parent.node.arguments.includes(fnInfo.path.node)) {
    return true;
  }
  return false;
}

function analyze(code) {
  const ast = parse(code);
  if (ast.__parseError) {
    return { parseError: ast.__parseError, violations: [], asyncDecls: [] };
  }

  // asyncDecls kept for the `asyncDecls` field in output (informational
  // only). Crash-risk detection now uses scope-aware path.scope.getBinding
  // via isAsyncByBinding inside calleeToName.
  const asyncDecls = discoverAsyncDecls(ast);
  const violations = [];

  traverse(ast, {
    CallExpression(path) {
      const name = calleeToName(path.node.callee, path);
      if (!name) return;
      if (isAwaited(path)) return;

      const fnInfo = enclosingFunction(path);
      const inSyncCb = isSyncCallback(fnInfo);
      const line = path.node.loc ? path.node.loc.start.line : null;
      const col  = path.node.loc ? path.node.loc.start.column : null;
      const thenCatch = hasThenCatch(path);

      // The CRASH-CAUSING pattern: unawaited async call inside a sync callback.
      // (with or without .then/.catch — concurrency happens either way.)
      if (inSyncCb) {
        violations.push({
          severity: "crash_risk",
          kind: thenCatch ? "sync_cb_fire_forget_then_catch" : "sync_cb_fire_forget_raw",
          name, line, col,
          enclosing_fn_async: fnInfo ? fnInfo.async : null,
          message: `unawaited async call '${name}' inside a synchronous callback — ` +
                   `causes unbounded concurrent ${name} invocations when the callback ` +
                   `is fired repeatedly (e.g. setInterval, exploreUntil, event handlers). ` +
                   `Fix: move the async call out of the callback into an async outer loop.`,
        });
      }
      // Unhandled-rejection warning only (no sync-callback) — log but not crash-risk
      else if (!thenCatch) {
        violations.push({
          severity: "warning",
          kind: "unhandled_rejection",
          name, line, col,
          enclosing_fn_async: fnInfo ? fnInfo.async : null,
          message: `unhandled promise from async '${name}' — consider adding await, ` +
                   `.catch, or Promise.all.`,
        });
      }
    },
  });

  return {
    parseError: null,
    asyncDecls: [...asyncDecls],
    violations,
    has_crash_risk: violations.some(v => v.severity === "crash_risk"),
  };
}

// CLI entry
if (require.main === module) {
  const codePath = process.argv[2];
  if (!codePath) {
    console.error("usage: node _async_validator.js <code_file.js>");
    process.exit(2);
  }
  const fs = require("fs");
  const code = fs.readFileSync(codePath, "utf8");
  const result = analyze(code);
  console.log(JSON.stringify(result));
}

module.exports = { analyze };
