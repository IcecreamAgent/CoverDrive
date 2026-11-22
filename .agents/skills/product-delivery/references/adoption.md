# Adapting product-delivery to another project

Start project-local. Promote stable, genuinely shared behavior to a personal Skill only after more than one repository has passed realistic trials.

## Portable layer

Copy and adapt:

- `.agents/skills/product-delivery/`
- `scripts/agent/`, including the candidate observer, validator, evidence recorder, `run-check.mjs`, and fresh-run replay runner `ci-verify.mjs`
- the `agent:validate`, `agent:check`, `agent:evidence`, `agent:ci`, and `agent:test` package/task-runner entry points
- empty `tasks/` and `runs/` directories and their ignore/retention convention; do not copy another project's task state or evidence

Custom Agent definitions can also be copied when their responsibilities and tools still match. Do not copy product facts, commands, approvals, credentials, acceptance criteria, or platform-specific release instructions.

## Minimum migration path

Use this sequence for a Web, app, or API repository. The repository's main Codex task owns the migration and remains the only orchestrator.

1. **Inventory the target.** Confirm the repository root, current working-tree state, real build/test commands, supported environments, release owner, and actions that still require human authorization. Do not begin by copying the source project's `PROJECT_PROFILE.md`, `AGENTS.md`, tasks, runs, or product documents.
2. **Install the portable runtime.** Copy `.agents/skills/product-delivery/` and `scripts/agent/`. Create empty `tasks/` and `runs/` directories. Copy `.codex/agents/` only for roles the target actually needs, then update their context paths and commands.
3. **Expose all contract commands.** Add equivalent task-runner entries for validation, real command execution, manual evidence, and contract tests. For a Node repository, the minimum mapping is:

   ```json
   {
     "scripts": {
       "agent:validate": "node scripts/agent/validate-task.mjs",
       "agent:check": "node scripts/agent/run-check.mjs",
       "agent:evidence": "node scripts/agent/record-evidence.mjs",
       "agent:ci": "node scripts/agent/ci-verify.mjs",
       "agent:test": "node --test scripts/agent/task-contract.test.mjs scripts/agent/agent-cli.test.mjs"
     }
   }
   ```

   Keep `agent:check` as the only local route for claiming that an automated acceptance command passed: it must run the command, capture its exit status, and write evidence with artifact digests. `agent:evidence` preserves sourced human review, user research, or other non-automated observations; keep those fail-closed for acceptance until the target has a trusted-source adapter. `agent:ci` can first be used locally as a fresh-run smoke test; enabling it as a protected merge check is a separate integration step.
4. **Describe this project, not the template project.** Create `PROJECT_PROFILE.md` from the target repository. Update `AGENTS.md` with its invariants, actual commands, UX gates, delegation rules, evidence requirements, and authorization boundaries. Route durable product, UX, architecture, operations, and release documents from the profile instead of embedding changing facts in Agent prompts.
5. **Map the delivery lifecycle.** Record the real install, local start, targeted test, full verification, build, preview/staging, deploy, health-check, rollback, and recovery commands. Automated acceptance criteria store exact argv arrays and bounded `timeout_seconds`, not shell prose. Mark unavailable capabilities explicitly. Add the platform-specific evidence from the table below. Keep `released` disabled until a target-specific adapter can query real authorization, deployed revision, environment, and post-release health evidence.
   Review exact generated-file exclusions in `candidate-revision.mjs`. This project uses a Vite/React frontend and Python backend, so only local secrets, caches, logs, `frontend/node_modules/`, `frontend/dist/`, task state, and run evidence are excluded. The matching regressions live in `agent-cli.test.mjs`. Never exclude product source, tests, configuration, models, migrations, or lockfiles merely to preserve old evidence.
6. **Validate the contract layer.** Run the target equivalents of `npm run agent:test` and `npm run agent:validate`. At this point there may be no accepted task, so an `agent:ci` invocation could replay zero criteria and proves nothing. Fix integration errors before using the workflow for product work; do not weaken the schema or evidence rules to make the checks pass.

   Preserve attack and drift regressions when porting: reject substituted argv; reject candidate, task-contract, or criterion changes during execution; reject old or caller-invented CI run IDs; detect changed artifacts; keep task/evidence-only commits candidate-stable; and reject task/runs symlink escapes.
7. **Run one real pilot.** Create `tasks/TASK-0001.md` from `assets/task-template.md`, define at least one observable acceptance criterion with an exact `command` argv array, and execute its automated check through the target equivalent of:

   ```bash
   npm run agent:check -- \
     --task tasks/TASK-0001.md \
     --criterion AC-01 \
     --summary "observable result"

   # Copy candidate_revision from the generated evidence into the task,
   # make the state-only transition to status: accepted, then validate.
   npm run agent:validate

   # Re-execute every required automated criterion under one fresh local run.
   npm run agent:ci -- --require-replay
   ```

   The full pilot sequence is `ready/in_validation → agent:check → bind the observed candidate → accepted → agent:validate → local fresh-run replay`. Confirm that the evidence is attached to the observed candidate revision and acceptance definition, that every required criterion was replayed, and that changing either makes stale evidence unusable. The local replay is deliberately labelled `local-ci-mode`; it validates the mechanism but is not proof from a remote CI provider.
8. **Expand only after the pilot.** Add specialists, CI triggers, preview deployments, monitoring, or issue-tracker integration only when a real routing or continuation need appears. Keep production release and other external writes behind the target project's explicit authorization boundary.

The migration is minimally complete when a fresh Codex session can read `AGENTS.md` and `PROJECT_PROFILE.md`, resume the pilot from `tasks/`, execute a real check, bind and validate an accepted candidate, complete a non-empty local replay, find its evidence under `runs/`, reject stale evidence, and state exactly which action still needs human authorization.

## Optional remote CI gate

Remote CI is not required for the first local pilot. When the project is ready to make it a merge gate, configure the provider to install from the lockfile, run `agent:ci -- --require-ci --require-replay`, run the normal lint/type/test/build checks, and finish with `agent:validate`. Make that job a protected required check. The runner records provider provenance and fails when `--require-ci` is used outside the supported trusted environment; a local replay must never be reported as remote CI success.

Use least-privilege test credentials and no production deployment, billing, migration, or destructive-data permissions. If a criterion needs an external system, give it an isolated non-production environment and record that environment in the task.

## Project onboarding

After the minimum migration works, harden it with three representative trials:

1. A bug, including one real failed check followed by a successful correction.
2. A user-facing feature, including the applicable interaction contract and a clear distinction between automated, Agent-review, real-user, and production evidence.
3. A preview or staging release, including one interrupted resume, one stale event, a health check, and a verified rollback path.

Preserve these cases as workflow evaluations. Select the smallest Agent pool that can run them: main Agent plus implementation, QA, and independent review is usually enough. Add product, UX, architecture, security, data, or release specialists only when routing needs them.

## Platform adaptations

| Project type | Additional execution and evidence |
| --- | --- |
| Web | Browser/device matrix, accessibility, preview URL, server logs, health check and rollback |
| Mobile app | Simulator and real-device targets, permissions, offline/background behavior, signed builds, store workflow and staged rollout |
| API/service | Contract compatibility, authentication/authorization, migrations, consumer tests, load/health signals and rollback |
| Desktop | Supported OS/build matrix, installer/signing, filesystem permissions, upgrade/uninstall behavior |
| Data/ML | Dataset and model versions, reproducibility, evaluation thresholds, data quality, drift and rollback |

When the repository has no Node.js task runner, either retain Node as a small developer-tool dependency for these scripts or port the candidate provider, validator, automated-check runner, evidence recorder, fresh-run replay runner, and contract tests to the repository's native language. Expose equivalent entry points through Make, Taskfile, Gradle, Xcode, Cargo, or the native build system. The contract matters; the command name does not.

The runtime observes Git worktrees through Git clean-filter semantics and uses a project-specific deterministic filesystem manifest when the repository is a plain directory. The filesystem provider hashes normalized paths, file type and raw bytes; it does not follow symlinks and rejects special files. Moving this project into Git intentionally changes the provider prefix and invalidates older evidence. Validate sparse-checkout, submodule, LFS and filter behavior before relying on a new Git setup. Never restore the unsafe behavior of trusting a caller-supplied candidate when observation fails.

## What requires extra infrastructure

Task files and a Skill support session and cross-session recovery. Background continuation requires an actual trigger such as CI, a scheduled automation, webhook receiver, issue tracker, or monitoring connector. Production isolation requires real credentials, branch protection, environment rules, and platform IAM; prompt instructions alone do not enforce it.
