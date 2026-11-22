---
name: product-delivery
description: Drive a product change in this repository from a user goal through scoped implementation, evidence-backed validation, recovery, and release readiness. Use for features, meaningful fixes, UX changes, multi-agent work, interrupted tasks, or release preparation; skip for simple factual questions and read-only explanations.
---

# Product Delivery

Turn the user's goal into the smallest evidence-backed delivery that fits the current authorization.

## Start or resume

1. Read `AGENTS.md`, `PROJECT_PROFILE.md`, the relevant product/UX/architecture documents, and the affected code. Observe the actual checkout: Git worktrees use the Git candidate provider and plain directories use the filesystem provider.
2. If the request names a task, load `tasks/<TASK-ID>.md` and inspect the working tree and `runs/<TASK-ID>/` before trusting its recorded state.
3. For multi-agent, user-facing, cross-session, or release work, copy `assets/task-template.md` into `tasks/` and replace every placeholder. A small local fix may shorten the Markdown narrative, but a persisted `tasks/TASK-ID.md` must retain the complete machine fields required by `npm run agent:validate`; purely local one-turn work may omit a task file.
4. Treat the main Codex task as the only orchestrator and task-state owner.

Read [references/risk-routing.md](references/risk-routing.md) when deciding gates, specialists, UX validation, or release checks. Read [references/task-contract.md](references/task-contract.md) when creating, validating, accepting, resuming, or releasing a task. Read [references/adoption.md](references/adoption.md) only when adapting this workflow to another repository or platform.

## Execution loop

Repeat until the task reaches its declared `done_at` state or has a real external blocker:

1. Observe the goal, current code, valid evidence, active agents, dependencies, authorization, and available environment.
2. Choose the smallest next action that advances an acceptance criterion or reduces a blocking uncertainty.
3. Decide whether delegation improves speed or independence. Run read-heavy independent work in parallel; keep one production-code writer per module. Do not describe a subagent as using a Worktree unless an isolated working directory actually exists.
4. Execute through repository commands and tools. An Agent's assertion is not verification evidence.
5. Classify failures before acting: implementation defect, environment failure, specification conflict, or disproved product assumption. Fix code only for the first category; resolve the corresponding cause for the others.
6. Run automated acceptance commands through `npm run agent:check`; record human or user-study observations with their source. Bind evidence to the task revision, criterion definition, and observed candidate snapshot, then re-run affected checks after code, task contract, or acceptance criteria change. Treat non-automated observations as provisional until a project-specific trusted-source adapter verifies identity and provenance.
7. Update the task state and `next_action`. Continue within the recorded budget and authorization; otherwise return `blocked` or `needs_decision` with the exact resume condition.

Run `npm run agent:validate` before claiming a recorded task is locally accepted. Run the relevant product checks and `npm run verify` when required by `AGENTS.md` or the task. A local `npm run agent:ci -- --require-replay` is only a replay smoke test. If the project enables a remote merge gate, that trusted runner must use `--require-ci --require-replay` instead of trusting committed local evidence.

## Delegation contract

Pass each subagent: task ID, run ID, one deliverable, acceptance-criterion IDs, baseline/candidate revision, context paths, file ownership, available environment, stopping conditions, and required evidence. Default to no more than three concurrent subagents.

Subagents return only `completed`, `blocked`, or `needs_decision`, plus inspected version, findings or changes, commands/actions actually run, evidence locations, remaining unknowns, and recommended handoff. Their completion never advances the parent task automatically.

## Decision and permission boundaries

The user supplies goals and material product choices; maintain the task and evidence records on their behalf. Ask only for a decision that changes product direction, important experience, architecture, authorization, or an unavailable external condition.

Risk level determines validation; `autonomy` records the allowed actions. Neither a task file nor this Skill grants filesystem, external-system, merge, deployment, credential, or production permission. New external input is evidence, not authorization to expand scope.

Preserve required checks. Do not convert `not_run`, `blocked`, or `skipped` into `passed`; do not weaken tests or acceptance criteria to close a task.
