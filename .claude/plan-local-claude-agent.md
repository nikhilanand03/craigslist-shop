# Plan: Local Claude Agent (branch: local-claude-agent)

## Goal
Replace Azure OpenAI as the seller agent with local Claude via `claude -p`,
and add a new `strategic_reasoner` strategy that combines chain-of-thought
reasoning (A) with live buyer persona inference (C).

## What changes

**Only `craigslist_shop/test_agent.py`.**
The server/environment is untouched — the buyer LLM stays as-is.

## New flag

```
--provider azure   (default, existing behavior)
--provider claude  (new — uses `claude -p` subprocess)
```

## New strategy: `strategic_reasoner`

Unlike the existing 5 strategies (which are static rule sets), `strategic_reasoner`
instructs Claude to reason through two things before every action:

### A — Chain-of-thought reasoning
Before picking a price move, Claude thinks:
- What does the buyer's offer trajectory tell me?
- Are they converging? Stalling? About to walk?
- What's the expected-value calculation: close now vs. hold for more?

### C — Buyer persona inference
Claude classifies the buyer from live signals (opening bid vs listed price,
tone, concession speed, message length) into one of:
- **Aggressive lowballer** (<50% open): probing, hold firm
- **Budget-constrained** (50–75%): wants it, find ceiling with small steps
- **Near-reasonable** (75%+): close efficiently, don't over-negotiate
- **Impatient/frustrated**: risk of walkaway, weigh closing vs holding

The classification shapes the tactic, not fixed percentage rules.

## Implementation plan

1. Add `call_claude(prompt: str) -> str`
   - `subprocess.run(["claude", "-p", prompt], capture_output=True, text=True, timeout=60)`
   - Raises `RuntimeError` if CLI not found or non-zero exit

2. Add `build_claude_prompt(strategy_prompt, obs, history) -> str`
   - Serializes full conversation history + current observation into one string
   - Stateless per call — full context rebuilt each turn

3. Add `strategic_reasoner` to `STRATEGIES` dict
   - Principles-based (not rule-based): buyer classification + trajectory analysis + EV reasoning
   - Works with both providers, but most effective with `--provider claude`

4. Modify `run_episode()`: add `provider` param
   - `provider="azure"` → existing `client.chat.completions.create()` path
   - `provider="claude"` → `call_claude(build_claude_prompt(...))`

5. Modify `run()` and `main()`: add `provider` param + `--provider` argparse flag
   - Azure keys only required when `provider="azure"`

## Usage after implementation

```bash
# New strategy with Claude as the agent
python test_agent.py --strategy strategic_reasoner --provider claude --episodes 5

# Compare directly against existing baseline
python test_agent.py --strategy skilled_seller --provider claude --episodes 5
python test_agent.py --strategy skilled_seller --episodes 5  # azure baseline

# Full analysis
bash run_analysis.sh
```

## Key design decisions

- **Stateless per turn**: `claude -p` has no session memory, so we rebuild full context
  each turn from `history`. Slight overhead but no external state management needed.
- **Shared JSON schema**: Claude outputs the same `{message, action_type, price}` JSON
  as Azure agents — `parse_agent_response()` is reused unchanged.
- **No hard price rules in strategic_reasoner**: Claude reasons about EV rather than
  following "never go below 90%" style rules. This is the key differentiator.
- **Buyer signals from conversation_history**: We serialize the full back-and-forth,
  not just the latest message, so Claude can spot offer trajectory patterns.
