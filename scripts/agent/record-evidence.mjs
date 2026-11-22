#!/usr/bin/env node

import { randomUUID } from "node:crypto";
import { realpath } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import {
  EVIDENCE_STATES,
  loadTask,
  validateTaskData,
} from "./task-contract.mjs";
import { currentCandidate } from "./candidate-revision.mjs";
import { writeEvidenceRecord } from "./evidence-writer.mjs";
import { safePathSegment } from "./path-safety.mjs";

function parseArgs(argv) {
  const result = { artifact: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) throw new Error(`unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`missing value for --${key}`);
    index += 1;
    if (key === "artifact") result.artifact.push(value);
    else result[key] = value;
  }
  return result;
}

function requireString(args, key) {
  const value = args[key];
  if (typeof value !== "string" || value.trim() === "") throw new Error(`--${key} is required`);
  return value;
}

try {
  const args = parseArgs(process.argv.slice(2));
  const projectRoot = process.cwd();
  const taskPath = path.resolve(projectRoot, requireString(args, "task"));
  const projectRealPath = await realpath(projectRoot);
  const taskRealPath = await realpath(taskPath);
  if (!taskRealPath.startsWith(`${projectRealPath}${path.sep}`)) throw new Error("task file must be inside the project");
  const criterionId = requireString(args, "criterion");
  const status = requireString(args, "status");
  const action = requireString(args, "action");
  const summary = requireString(args, "summary");

  if (!EVIDENCE_STATES.has(status)) throw new Error(`unsupported evidence status: ${status}`);

  const { data: task } = await loadTask(taskRealPath);
  const taskErrors = validateTaskData(task);
  if (taskErrors.length > 0) throw new Error(`task contract is invalid: ${taskErrors.join("; ")}`);
  const criterion = task.acceptance_criteria?.find((item) => item.id === criterionId);
  if (!criterion) throw new Error(`${criterionId} is not defined by ${task.id}`);
  if (criterion.verification_type === "automated") {
    throw new Error("automated evidence can only be created by npm run agent:check");
  }

  const observedCandidate = await currentCandidate(projectRoot);
  if (args.candidate && args.candidate !== observedCandidate) {
    throw new Error(`requested candidate ${args.candidate} does not match observed candidate ${observedCandidate}`);
  }
  const candidate = observedCandidate;
  if (task.candidate_revision && task.candidate_revision !== candidate) {
    throw new Error(`candidate ${candidate} does not match task candidate_revision ${task.candidate_revision}`);
  }
  const exitCode = args["exit-code"] === undefined ? null : Number(args["exit-code"]);
  if (exitCode !== null && !Number.isInteger(exitCode)) throw new Error("--exit-code must be an integer");
  if (status === "passed" && !args.source) {
    throw new Error(`${criterion.verification_type} passed evidence requires --source`);
  }
  if (status === "skipped" && !args.reason) throw new Error("skipped evidence requires --reason");

  const runId = safePathSegment(args["run-id"] ?? `RUN-${new Date().toISOString().replace(/[-:.TZ]/g, "")}-${randomUUID().slice(0, 8)}`, "run ID");
  const { target } = await writeEvidenceRecord({
    projectRoot,
    task,
    criterion,
    runId,
    candidate,
    status,
    environment: args.environment ?? "local",
    action,
    argv: null,
    exitCode,
    runner: null,
    summary,
    source: args.source ?? null,
    reason: args.reason ?? null,
    artifacts: args.artifact,
  });
  console.log(path.relative(projectRoot, target));
  console.log(`candidate_revision=${candidate}`);
} catch (error) {
  console.error(`Unable to record evidence: ${error.message}`);
  process.exitCode = 1;
}
