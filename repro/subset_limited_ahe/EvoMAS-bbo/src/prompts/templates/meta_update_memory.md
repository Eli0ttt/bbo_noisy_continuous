# Meta Model - Update Memory Action

You are analyzing the outcome of a meta-model action to extract learning experiences.

## Context

The meta-model performs three types of actions to evolve Multi-Agent Systems (MAS):
- **Generate**: Adapt MAS configuration to current tasks
- **Mutate**: Modify components based on execution observations
- **Crossover**: Combine two parent configurations

After each action, we evaluate the new configuration on a minibatch and compare it to the previous performance.

## Your Task

Analyze the action taken and its outcome to extract insights about what made the configuration change successful or unsuccessful.

## Input

**Action Type:** {{action_type}}

**Query/Task:** {{query}}

**Previous Configuration:**
```yaml
{{old_config}}
```

**New Configuration:**
```yaml
{{new_config}}
```

**Performance Change:**
- Previous Accuracy: {{old_accuracy}}
- New Accuracy: {{new_accuracy}}
- Result: {{result}}

**Configuration Changes:**
{{config_changes}}

## Your Goal

Analyze this experience and extract actionable insights:

1. **What Changed**: Identify the key modifications made to the configuration
2. **Why It Succeeded/Failed**: Determine which changes contributed to the performance change
3. **Patterns**: Identify patterns that can guide future actions
4. **Lessons Learned**: What should we remember for similar situations?

## Output Format

Provide your analysis in the following structure:

**Key Changes:**
[Summarize the main configuration changes made - focus on what was modified, not just that something changed]

**Success/Failure Analysis:**
[Explain WHY the changes led to better or worse performance. Be specific about which aspects of the changes were beneficial or harmful]

**Insights for Future Actions:**
[What should we learn from this experience? When should we apply similar changes? When should we avoid them?]

**Action Experience Summary:**
[A concise 2-3 sentence summary capturing the most important learning from this experience]

Keep your analysis clear and focused on actionable insights that can guide future configuration generation.
