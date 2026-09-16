// BBO visible-selfcheck routing guard.
//
// This is harness policy, not task code. It does not alter /app/selfcheck.py.
// It prevents the model-facing Bash tool from directly executing selfcheck.py;
// visible scoring must go through version_checkpoint.py baseline/evaluate so
// every score used for research has auditable checkpoint state.

export const name = 'bbo-selfcheck-guard'
export const inject = ['tools']

export const DENIAL =
  'BBO_DIRECT_SELFCHECK_DENIED: run visible scoring through ' +
  '`python /opt/dsh-config/version_checkpoint.py baseline` for v0 or ' +
  '`python /opt/dsh-config/version_checkpoint.py evaluate --version vN --description "..."` ' +
  'for a candidate.'

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
      if (exec.name !== 'bash') return undefined
      const args = exec.arguments
      if (args === null || typeof args !== 'object') return undefined
      const command = args.command
      if (isDirectSelfcheckCommand(command)) return DENIAL
      return undefined
    }),
    'bbo-selfcheck-guard',
  )
}
