# Agent Execution Contract

This repository uses Selective Intelligence (SI) for all substantial agent work.

Canonical skill: `.agents/skills/selective-intelligence/SKILL.md`.
Continuity rules: `.agents/skills/selective-intelligence/references/continuity-and-impact.md`.

## Resume-first requirement

A new Work, Codex, or other agent session MUST recover the current breakpoint before broad inspection. Treat an interrupted project as existing work, not a fresh project.

1. Read this file and the canonical SI skill first.
2. Locate the active branch/PR, checkpoint, resume packet, queue item, or equivalent persisted state.
3. Inspect only the files, owners, tests, and evidence needed for the next unproven transition.
4. Reuse still-valid evidence. Do not repeat repository-wide audits or full-suite validation merely because the agent/session changed.
5. Parallel lanes must share current architecture, ownership, dependencies, and completed evidence rather than rediscovering them independently.
6. Use targeted validation during bounded implementation. Run broad regression only when shared behavior changed, evidence was invalidated, or an integration/release boundary requires it.
7. Before interruption, handoff, capacity exhaustion, or context switch, persist a resumable checkpoint containing exact completed work, changed-but-unverified work, tests/evidence, branch/revision, blockers, actions that must not be repeated, and one next safe action.
8. `Keep going` means continue from that checkpoint; it does not authorize another deep dive from zero.

Optimize for verified implementation completed per unit of compute. Reasoning, retrieval, review, and testing that do not change a decision, reduce risk, or prove the requested outcome are waste and should be omitted.
