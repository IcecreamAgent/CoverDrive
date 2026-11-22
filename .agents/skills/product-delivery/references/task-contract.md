# Task and evidence contract

Task files live at `tasks/TASK-*.md`. Their front matter is JSON-compatible YAML so the repository can validate it without adding a parsing dependency. The Markdown body holds rationale and human-readable notes; the front matter is the current machine-checkable state.

The current task and evidence schema is version 2. Version 1 evidence cannot move a task into `accepted` or `released`.

## States

`proposed → ready → in_progress → in_validation → accepted → released`

- `blocked` records an unavailable external condition and a concrete `resume_condition`.
- `needs_decision` records the decision, evidence, options, and recommended path.
- Work without release ends at `accepted`; work whose `done_at` is `released` is incomplete at `accepted`.

Only the main Agent changes the parent task state. A subagent status is an input, not a state transition.

## Transition rules

- `ready`: scope and versioned acceptance criteria are clear; dependencies, environment, authorization, and next action are known.
- `in_progress`: executor, actual workspace, baseline and owned files/resources are assigned.
- `in_validation`: an identifiable candidate exists and implementation self-checks are recorded.
- `accepted`: every required criterion has valid evidence for the task revision, candidate snapshot and criterion definition; required review has no unresolved blocker; the final integrated candidate passed the required project checks. The portable contract fails closed for required human or user-study criteria until a trusted-source adapter exists.
- `released`: fail closed in the portable contract. Enable this state only after a project-specific adapter verifies a structured authorization record, the target environment's deployed revision, and real post-release health evidence. Preserve release history separately from current deployment health.

If candidate code or an acceptance criterion changes, invalidate affected evidence. An unpublished accepted task returns to `in_validation` or `in_progress`. A released version remains historical; follow-up repair uses a linked task.

## Evidence

Automated checks must run through the command wrapper so the evidence records the observed exit code and log:

Define the exact argv and timeout on the criterion, for example `"command": ["npm", "test", "--", "tests/example.test.ts"]` and `"timeout_seconds": 600`. Do not store the command only as prose or replace it at execution time. A timeout terminates the process group where supported, records exit code 124 and failed evidence, and never counts as passed.

```text
npm run agent:check -- \
  --task tasks/TASK-0001.md \
  --criterion AC-01 \
  --summary "Expected behavior observed"
```

For `human_review` or `user_study`, use `npm run agent:evidence` and pass `--source` with the actual record or artifact reference. This preserves an observation but does not verify identity or provenance. A required non-automated criterion remains unable to satisfy `accepted` until a project-specific adapter validates a trusted review, approval, or research source. The tools write under `runs/<TASK-ID>/<RUN-ID>/` and record task revision, criterion definition digest, candidate snapshot, environment, action, exit code, time, result, artifact references, and artifact SHA-256 digests.

The wrapper freezes the candidate, task revision, criterion digest, and declared argv before execution and checks them again afterward. It writes a project-local `.txt` artifact; missing, changed, or escaping artifacts invalidate evidence. Local records are auditable consistency data, not cryptographic attestation: a caller that controls the machine can tamper with the runner or environment. `npm run agent:ci -- --require-replay` exercises the same fresh-run rule locally and labels its evidence `local-ci-mode`; it is not proof that a remote CI provider ran. When a project enables a trusted CI merge gate, use `--require-ci --require-replay` there. Do not run commands that print credentials or sensitive payloads because command output is stored as evidence.

Allowed evidence statuses: `passed`, `failed`, `blocked`, `not_run`, and `skipped`. A skip needs an applicability or approved-exception reason and never counts as passed.

Run `npm run agent:validate`. Files directly under `tasks/` are active delivery tasks: `in_validation`, `accepted`, and `released` tasks must also name the currently observed project candidate. Validation fails if required evidence is missing, stale, superseded by a later failure, or belongs to another task revision, candidate, or criterion definition. Changing contract-bearing task fields requires incrementing `revision` and rerunning affected evidence; a state-only transition may keep the same revision. Move completed historical records to `tasks/archive/`; default validation intentionally ignores that archive rather than pretending an old candidate is the current worktree.

## Recovery

On resume, inspect actual Git/worktree state, active agents, current environment, external operation state, and evidence before using the recorded `next_action`. Use a unique run ID and do not repeat a timed-out external mutation until its real status has been checked. Ignore duplicate or old events that reference another task revision or candidate.
