// Split-collaboration guard. The coordinator, not either model, owns scoring,
// merge, the canonical solver, and raw traces.
export const name = 'bbo-split-guard'
export const inject = ['tools']

const main = /\/app\/methods\/main\//
const role = String(process.env.SPLIT_ROLE || '')
const ownArtifact = (raw) => {
  if (!/^[AB]$/.test(role)) return false
  const re = new RegExp('/app/methods/main/\\.collab/versions/v\\d+/' + role + '/(?:candidate\\.py|handoff\\.json)')
  return re.test(raw)
}
const scoring = /\bselfcheck\.py\b/i
const traces = /(?:\/logs\/artifacts\/bbo-split\/traces|\/tmp\/bbo-split-sessions|session\.jsonl(?:\.zstd)?)/i

export function apply(ctx) {
  ctx.effect(
    () => ctx.tools.guard((exec) => {
      if (!exec || !exec.arguments) return undefined
      const raw = JSON.stringify(exec.arguments)
      if (traces.test(raw)) {
        return 'Raw DSH traces are coordinator-owned. Work from parent.py, contract.json and your candidate only.'
      }
      if (exec.name === 'bash') {
        const command = String(exec.arguments.command || '')
        if (scoring.test(command)) {
          return 'The coordinator owns visible selfcheck. Do not run selfcheck.py.'
        }
        if (main.test(command) && !ownArtifact(command) &&
            /(?:>|>>|\bcp\b|\bmv\b|\brm\b|\bsed\b|\btee\b|\bpython\b|\bpython3\b)/i.test(command)) {
          return 'Do not mutate canonical /app/methods/main or peer artifacts. Edit only your own candidate.py and handoff.json.'
        }
      }
      if (['write','edit','delete','remove','move','rename'].includes(exec.name) && main.test(raw) && !ownArtifact(raw)) {
        return 'Do not mutate canonical /app/methods/main or peer artifacts. Edit only your own candidate.py and handoff.json.'
      }
      return undefined
    }),
    name,
  )
}
