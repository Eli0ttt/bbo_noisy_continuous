// BBO visible-selfcheck and version-ledger routing guard.
//
// This is harness policy, not task code. It does not alter /app/selfcheck.py.
// It prevents the model-facing Bash tool from directly executing selfcheck.py;
// visible scoring must go through version_checkpoint.py baseline/evaluate so
// every score used for research has auditable checkpoint state.  It also keeps
// the version snapshot directory helper-owned: model shell code may inspect a
// snapshot read-only, but only version_checkpoint.py may create, overwrite,
// delete, or restore one.

export const name = 'bbo-selfcheck-guard'
export const inject = ['tools']

export const DENIAL =
  'BBO_DIRECT_SELFCHECK_DENIED: the harness records official visible scoring atomically. ' +
  'For v0 use `python /opt/dsh-config/version_checkpoint.py baseline`. ' +
  'For a candidate use `python /opt/dsh-config/version_checkpoint.py evaluate --version vN --description "..."`; ' +
  'then resolve that same pending version with `keep --version vN` or `revert --version vN`. ' +
  'The helper executes the unmodified official /app/selfcheck.py and writes the official experiment log/snapshot record.'

export const VERSION_MUTATION_DENIAL =
  'BBO_DIRECT_VERSION_MUTATION_DENIED: /app/methods/versions is an immutable, helper-owned ledger. ' +
  'Do not create, copy, overwrite, delete, move, chmod, or restore snapshots from Bash, Python, or another subprocess. ' +
  'Use only `python /opt/dsh-config/version_checkpoint.py baseline|evaluate|keep|revert`. ' +
  'Read-only inspection is allowed; this denial is recorded in the raw DSH trace and trace audit.'

// A DSH tool guard receives source text, not a parsed shell/Python AST. These
// patterns deliberately cover the ordinary mutation forms used in this task.
// The post-run trace audit independently re-detects the same class and fails
// closed if a mutation was not denied.  Pure inspection (ls, cat, hashing,
// importing a snapshot for a benchmark) remains allowed.
const VERSION_REFERENCE =
  /(?:\/app\/)?methods\/versions(?:\/|\b)|(?:^|[\s"'=;|&])(?:\.\/)?versions(?:\/|\b)|\bBBO_METHODS_ROOT\b/i

const MUTATING_SHELL_OPERATION =
  /(?:^|[\s;&|])(?:rm|rmdir|unlink|mv|cp|install|mkdir|mktemp|touch|tee|dd|truncate|ln|rsync|chmod|chown|tar|unzip|zip)(?:\s|$)|\bfind\b[\s\S]*\s-delete\b|\bsed\s+-[^\n]*i\b|\bperl\s+-[^\n]*i\b|(?:>|>>)\s*(?:(?:\/app\/)?methods\/versions|(?:\.\/)?versions)(?:\/|\b)/i

const INTERPRETER =
  /(?:^|[\s;&|])(?:\S*\/)?(?:python(?:3(?:\.\d+)?)?|pypy(?:3)?|node|perl|ruby|php|sh|bash)(?:\s|$)/i

const PROGRAM_MUTATION_CUE =
  /(?:\b(?:subprocess\.(?:run|call|check_call|check_output|Popen)|os\.system)\s*\([^\n]*?["'](?:rm|rmdir|unlink|mv|cp|install|mkdir|touch|tee|dd|truncate|ln|rsync|chmod|chown)\b|\bos\.(?:remove|unlink|rename|replace|mkdir|makedirs)|shutil\.(?:copy|copy2|copyfile|copytree|move|rmtree)|pathlib\.[A-Za-z_]+\.?(?:mkdir|unlink|rename|replace)|(?:write_text|write_bytes|touch|mkdir|unlink|rename|replace)\s*\(|open\s*\([^\n]*,\s*["'](?:w|a|x|r\+))/i

const VERSION_MUTATING_FS_TOOLS = new Set([
  'write', 'edit', 'delete', 'remove', 'move', 'rename', 'chmod',
])

export function isDirectVersionFilesystemMutation(exec) {
  if (!exec || !VERSION_MUTATING_FS_TOOLS.has(exec.name)) return false
  const args = exec.arguments
  if (args === null || typeof args !== 'object') return false
  // Do not constrain /app/methods/main or experiment_log.md: the official
  // task explicitly requires the agent to edit those. Only snapshots are
  // helper-owned and immutable.
  return VERSION_REFERENCE.test(JSON.stringify(args))
}

export function isDirectVersionMutationCommand(command) {
  if (typeof command !== 'string') return false
  if (!VERSION_REFERENCE.test(command)) return false

  // Interpreter snippets that explicitly perform a mutating operation are
  // denied. A guard cannot prove arbitrary external program source is
  // read-only, so the formal trace audit remains mandatory for provenance.
  return MUTATING_SHELL_OPERATION.test(command) || (
    INTERPRETER.test(command) && PROGRAM_MUTATION_CUE.test(command)
  )
}

export function isDirectSelfcheckCommand(command) {
  if (typeof command !== 'string') return false
  const text = command.toLowerCase()
  if (!text.includes('selfcheck.py')) return false

  // Source inspection is still allowed. The guard targets execution paths.
  // It is intentionally conservative for shell commands that combine a Python
  // interpreter (or common execution helper) with selfcheck.py.
  const executionMarker =
    /(^|[\s;&|])(?:\/[^\s;&|]+\/)?python(?:3(?:\.\d+)?)?(?=$|[\s;&|])/m.test(text) ||
    /(^|[\s;&|])(?:uv\s+run\s+)?python(?:3(?:\.\d+)?)?(?=$|[\s;&|])/m.test(text) ||
    /(^|[;&|]\s*)(?:\.\/|\/app\/)selfcheck\.py(?=$|[\s;&|])/m.test(text) ||
    /\brunpy\b|\bsubprocess\b|\bexec\s*\(|\bxargs\b/.test(text)

  return executionMarker
}

export function apply(ctx) {
  ctx.effect(
    () => ctx.tools.guard((exec) => {
      if (isDirectVersionFilesystemMutation(exec)) return VERSION_MUTATION_DENIAL
      if (exec.name !== 'bash') return undefined
      const args = exec.arguments
      if (args === null || typeof args !== 'object') return undefined
      const command = args.command
      if (isDirectSelfcheckCommand(command)) return DENIAL
      if (isDirectVersionMutationCommand(command)) return VERSION_MUTATION_DENIAL
      return undefined
    }),
    'bbo-selfcheck-guard',
  )
}
