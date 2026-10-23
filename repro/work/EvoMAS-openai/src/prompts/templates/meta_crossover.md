# Meta Model - Crossover Action

You are a meta model that evolves Multi-Agent Systems (MAS) through crossover operations.

## Task

Create a new MAS configuration by combining strengths of two parent configurations while avoiding their weaknesses.

## CRITICAL CONSTRAINT

**Topology Inheritance:**
- **You MUST inherit the ENTIRE topology (reports_to structure) from EXACTLY ONE parent**
- Choose which parent's topology to use based on which has better coordination/accuracy

**Agent Recombination:**
- For EACH agent position in the inherited topology, you can:
  - Take the agent configuration from Parent 1
  - Take the agent configuration from Parent 2
  - Create a modified version by combining properties from both parents' agents

**DO NOT:**
- Create a new topology structure from scratch
- Add or remove agent positions beyond what exists in the chosen parent's topology
- Change the 'name' or 'backend' fields

**Only update the 'description' field to reflect the crossover.**

## Input

**Parent MAS 1:**
```yaml
{{mas_config_1}}
```
- Accuracy: {{accuracy_1}}
- Strengths: {{strengths_1}}
- Weaknesses: {{weaknesses_1}}

**Parent MAS 2:**
```yaml
{{mas_config_2}}
```
- Accuracy: {{accuracy_2}}
- Strengths: {{strengths_2}}
- Weaknesses: {{weaknesses_2}}

**Execution Logs:**
{{execution_logs}}

## Your Goal

Combine the best aspects of both parents to create an offspring MAS:

1. **Topology Selection**: Choose which parent's topology structure to inherit (consider coordination efficiency)
2. **Agent Selection**: For each agent in the inherited topology, choose the best agent configuration
3. **Prompt Combination**: Merge or select best prompts for each agent
4. **Model Assignment**: Choose best models for each agent role
5. **Configuration Synthesis**: Balance complexity and performance

## Output Format

**Crossover Strategy:**
[Explain your combination strategy]

**Topology Choice:**
[Specify which parent's topology you are inheriting: Parent 1 or Parent 2, and why]

**Agent Selection:**
[For EACH agent in the inherited topology, specify:
- Agent ID: <agent_id>
- Source: Parent 1 / Parent 2 / Hybrid
- Justification: Why this choice is better
]

Example:
- worker: Parent 1 (better prompt for task understanding)
- aggregator: Hybrid (Parent 1's prompt + Parent 2's model)
- judge: Parent 2 (more effective evaluation logic)

**Offspring Configuration:**
```yaml
[Full offspring YAML configuration with inherited topology and selected/combined agents]
```

**Rationale:**
[Explain why this combination should outperform both parents]

{{model_constraint}}

{{memory_context}}
