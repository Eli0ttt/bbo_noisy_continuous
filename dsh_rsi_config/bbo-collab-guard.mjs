// Model-facing guard. Coordinator environment.exec does not use DSH tools.
export const name = 'bbo-collab-guard'
export const inject = ['tools']

const forbidden = /(?:\bselfcheck\.py\b|\/app\/methods\/main\/)/i
const mutating = /(?:\b(?:cp|mv|rm|sed|tee|touch|chmod|install|rsync|python(?:3)?|node|bash|sh)\b|>|\bwrite\b)/i
// Raw DSH session traces are coordinator-owned audit data. Models collaborate
// through the bounded review.md artifact; they must not inspect one another's
// private reasoning/event streams or their own raw trace files.
const privateTrace = /(?:\/logs\/artifacts\/bbo-collab\/traces(?:\/|\b)|\/tmp\/bbo-collab-r[0-9]+-sessions(?:\/|\b)|session\.jsonl(?:\.zstd)?\b|\.jsonl\.zstd\b)/i

export function apply(ctx) {
  ctx.effect(
    () => ctx.tools.guard((exec) => {
      if (!exec || !exec.arguments) return undefined
      if (privateTrace.test(JSON.stringify(exec.arguments))) {
        return 'Raw DSH session traces are coordinator-owned audit records.'
      }
      if (exec.name === 'bash') {
        const command = exec.arguments.command
        if (typeof command === 'string' && forbidden.test(command) && mutating.test(command)) {
          return 'The coordinator owns selfcheck and /app/methods/main/. Work only in your assigned scratch directory.'
        }
      }
      if (['write','edit','delete','remove','move','rename'].includes(exec.name)
          && /\/app\/methods\/main\//.test(JSON.stringify(exec.arguments))) {
        return 'Write only your isolated scratch directory; the coordinator installs the selected solver.'
      }
      return undefined
    }),
    name,
  )
}
