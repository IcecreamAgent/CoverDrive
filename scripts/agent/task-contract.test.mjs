import assert from "node:assert/strict";
import test from "node:test";
import {
  acceptanceCriterionDigest,
  taskContractDigest,
  validateAcceptanceEvidence,
  validateEvidenceRecord,
  validateTaskData,
} from "./task-contract.mjs";

function task(overrides = {}) {
  return {
    schema_version: 2,
    id: "TASK-TEST",
    revision: 1,
    goal: "Prove the contract",
    user_scenario: "A maintainer validates a candidate",
    scope: ["contract"],
    non_scope: [],
    done_at: "accepted",
    acceptance_criteria: [{
      id: "AC-01",
      version: 1,
      expected: "The check succeeds",
      verification_type: "automated",
      verification: "node -e process.exit(0)",
      command: ["node", "-e", "process.exit(0)"],
      timeout_seconds: 30,
      required: true,
    }],
    risk_level: "L1",
    risk_tags: [],
    autonomy: "implementation",
    context_refs: [],
    baseline: "base-1",
    dependencies: [],
    owner: "main-agent",
    write_boundary: [],
    execution_environment: {},
    human_gates: [],
    required_evidence: ["AC-01"],
    budget: { max_parallel_subagents: 3, same_failure_retries_before_replan: 2, time_limit_minutes: null },
    active_run: null,
    attempt: 1,
    candidate_revision: "candidate-1",
    release: { target_environment: null, authorization_ref: null, deployed_revision: null, status: null, evidence: [] },
    next_action: "Validate",
    resume_condition: null,
    status: "accepted",
    ...overrides,
  };
}

function evidence(sourceTask, overrides = {}) {
  const criterion = sourceTask.acceptance_criteria[0];
  return {
    schema_version: 2,
    task_id: sourceTask.id,
    task_revision: sourceTask.revision,
    task_contract_digest: taskContractDigest(sourceTask),
    run_id: "RUN-TEST",
    criterion_id: criterion.id,
    criterion_version: criterion.version,
    criterion_digest: acceptanceCriterionDigest(criterion),
    verification_type: criterion.verification_type,
    candidate_revision: sourceTask.candidate_revision,
    status: "passed",
    environment: "test",
    action: "node -e process.exit(0)",
    argv: criterion.command,
    exit_code: 0,
    runner: "agent-check-v1",
    summary: "Passed",
    source: null,
    reason: null,
    artifacts: ["runs/log.txt"],
    artifact_digests: { "runs/log.txt": "0".repeat(64) },
    created_at: "2026-09-20T10:00:00.000Z",
    ...overrides,
  };
}

test("accepts a complete task with current captured evidence", () => {
  const currentTask = task();
  const currentEvidence = evidence(currentTask);
  assert.deepEqual(validateTaskData(currentTask), []);
  assert.deepEqual(validateEvidenceRecord(currentEvidence), []);
  assert.deepEqual(validateAcceptanceEvidence(currentTask, [currentEvidence]), []);
});

test("a later failure supersedes an earlier pass", () => {
  const currentTask = task();
  const passed = evidence(currentTask);
  const failed = evidence(currentTask, {
    status: "failed",
    exit_code: 1,
    created_at: "2026-09-20T10:01:00.000Z",
  });
  assert.match(validateAcceptanceEvidence(currentTask, [passed, failed])[0], /latest current evidence is not passed/);
});

test("changing an acceptance definition invalidates old evidence", () => {
  const original = task();
  const oldEvidence = evidence(original);
  const changed = task({
    acceptance_criteria: [{ ...original.acceptance_criteria[0], expected: "A different result" }],
  });
  assert.match(validateAcceptanceEvidence(changed, [oldEvidence])[0], /latest current evidence is not passed/);
});

test("running a different command cannot satisfy an automated criterion", () => {
  const currentTask = task();
  const wrongCommand = evidence(currentTask, { argv: ["node", "-e", "process.exit(0)", "wrong"] });
  assert.match(validateAcceptanceEvidence(currentTask, [wrongCommand])[0], /latest current evidence is not passed/);
});

test("a different task revision invalidates old evidence", () => {
  const original = task();
  const oldEvidence = evidence(original);
  const revised = task({ revision: 2 });
  assert.match(validateAcceptanceEvidence(revised, [oldEvidence])[0], /latest current evidence is not passed/);
});

test("changing a task gate invalidates evidence even without a revision bump", () => {
  const original = task();
  const oldEvidence = evidence(original);
  const changed = task({ human_gates: ["security-owner approval"] });
  assert.match(validateAcceptanceEvidence(changed, [oldEvidence])[0], /latest current evidence is not passed/);
});

test("CI validation requires evidence from its fresh automated run", () => {
  const currentTask = task();
  const localEvidence = evidence(currentTask);
  assert.match(
    validateAcceptanceEvidence(currentTask, [localEvidence], { requiredAutomatedRun: "CI-FRESH" })[0],
    /CI-FRESH/,
  );
});

test("portable validation fails closed for unverifiable human evidence", () => {
  const currentTask = task({
    acceptance_criteria: [{
      id: "AC-01",
      version: 1,
      expected: "A real participant completes the task",
      verification_type: "user_study",
      verification: "Observe the participant",
      command: null,
      timeout_seconds: null,
      required: true,
    }],
  });
  const claimed = evidence(currentTask, {
    verification_type: "user_study",
    argv: null,
    exit_code: null,
    runner: null,
    source: "x",
  });
  assert.match(validateAcceptanceEvidence(currentTask, [claimed])[0], /trusted-source adapter/);
});

test("rejects incomplete evidence and unsafe release state", () => {
  const currentTask = task();
  assert.ok(validateEvidenceRecord({ status: "passed" }).length > 0);
  const released = task({ status: "released", done_at: "released" });
  assert.ok(validateTaskData(released).some((error) => error.includes("release.target_environment")));
  assert.ok(validateTaskData(released).some((error) => error.includes("project-specific release adapter")));
});
