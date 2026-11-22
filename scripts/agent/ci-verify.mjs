#!/usr/bin/env node

import { randomUUID } from "node:crypto";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { discoverTaskFiles, loadTask, validateTaskData } from "./task-contract.mjs";

const projectRoot = process.cwd();
const flags = new Set(process.argv.slice(2));
for (const flag of flags) {
  if (!new Set(["--require-ci", "--require-replay"]).has(flag)) {
    console.error(`Unsupported option: ${flag}`);
    process.exit(1);
  }
}

const githubActions = process.env.CI === "true"
  && process.env.GITHUB_ACTIONS === "true"
  && Boolean(process.env.GITHUB_RUN_ID)
  && Boolean(process.env.GITHUB_SHA)
  && Boolean(process.env.GITHUB_REPOSITORY);

if (flags.has("--require-ci") && !githubActions) {
  console.error("A trusted CI environment is required; this command is not running inside GitHub Actions.");
  process.exit(1);
}

const execution = githubActions
  ? {
      environment: "ci:github-actions",
      label: "CI",
      prefix: "CI",
      provenance: {
        provider: "github-actions",
        run_id: process.env.GITHUB_RUN_ID,
        run_attempt: process.env.GITHUB_RUN_ATTEMPT ?? null,
        workflow: process.env.GITHUB_WORKFLOW ?? null,
        repository: process.env.GITHUB_REPOSITORY,
        revision: process.env.GITHUB_SHA,
        ref: process.env.GITHUB_REF ?? null,
        event_name: process.env.GITHUB_EVENT_NAME ?? null,
        url: `${process.env.GITHUB_SERVER_URL ?? "https://github.com"}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`,
      },
    }
  : {
      environment: "local-ci-mode",
      label: "Local CI-mode",
      prefix: "LOCAL-CI",
      provenance: { provider: "local", mode: "ci-replay-smoke" },
    };

const runId = `${execution.prefix}-${new Date().toISOString().replace(/[-:.TZ]/g, "")}-${randomUUID().slice(0, 8)}`;
const encodedProvenance = Buffer.from(JSON.stringify(execution.provenance)).toString("base64url");
const taskFiles = await discoverTaskFiles(projectRoot);
let failed = false;
let replayedCriteria = 0;

for (const taskFile of taskFiles) {
  const { data: task } = await loadTask(taskFile);
  const errors = validateTaskData(task);
  if (errors.length > 0) {
    failed = true;
    console.error(`Cannot replay ${path.relative(projectRoot, taskFile)}: ${errors.join("; ")}`);
    continue;
  }
  if (!new Set(["accepted", "released"]).has(task.status)) continue;

  for (const criterion of task.acceptance_criteria.filter((item) => item.required && item.verification_type === "automated")) {
    replayedCriteria += 1;
    const result = spawnSync(process.execPath, [
      path.join(projectRoot, "scripts/agent/run-check.mjs"),
      "--task", path.relative(projectRoot, taskFile),
      "--criterion", criterion.id,
      "--run-id", runId,
      "--environment", execution.environment,
      "--provenance", encodedProvenance,
      "--summary", `${execution.label} replay for ${criterion.id}`,
    ], { cwd: projectRoot, encoding: "utf8", stdio: "inherit" });
    if (result.status !== 0) failed = true;
  }
}

if (flags.has("--require-replay") && replayedCriteria === 0) {
  failed = true;
  console.error("No accepted or released automated acceptance criteria were available to replay.");
}

if (!failed) {
  const validation = spawnSync(process.execPath, [
    path.join(projectRoot, "scripts/agent/validate-task.mjs"),
    "--require-automated-run", runId,
  ], { cwd: projectRoot, encoding: "utf8", stdio: "inherit" });
  if (validation.status !== 0) failed = true;
}

if (failed) {
  console.error(`${execution.label} acceptance replay failed (${runId}).`);
  process.exitCode = 1;
} else {
  console.log(`${execution.label} acceptance replay passed (${runId}; ${replayedCriteria} criteria).`);
  if (!githubActions) console.log("This local smoke replay does not satisfy the remote required CI check.");
}
