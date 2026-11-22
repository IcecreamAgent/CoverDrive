#!/usr/bin/env node

import { randomUUID } from "node:crypto";
import { realpath, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { spawn, spawnSync } from "node:child_process";
import {
  acceptanceCriterionDigest,
  commandsEqual,
  loadTask,
  taskContractDigest,
  validateTaskData,
} from "./task-contract.mjs";
import { currentCandidate } from "./candidate-revision.mjs";
import { writeEvidenceRecord } from "./evidence-writer.mjs";
import { ensureProjectDirectory, safePathSegment } from "./path-safety.mjs";

const MAX_CAPTURE_BYTES = 10 * 1024 * 1024;

function terminateProcessTree(child, signal) {
  if (!child.pid) return;
  if (process.platform === "win32") {
    const args = ["/PID", String(child.pid), "/T"];
    if (signal === "SIGKILL") args.push("/F");
    spawnSync("taskkill", args, { windowsHide: true, stdio: "ignore" });
    return;
  }
  if (process.platform !== "win32") {
    try {
      process.kill(-child.pid, signal);
      return;
    } catch {
      // Fall back to the direct child when the process group is already gone.
    }
  }
  try {
    child.kill(signal);
  } catch {
    // The child may have exited between the timer and this call.
  }
}

function executeCommand(command, cwd, timeoutMs) {
  return new Promise((resolve) => {
    const child = spawn(command[0], command.slice(1), {
      cwd,
      detached: process.platform !== "win32",
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    const stdoutChunks = [];
    const stderrChunks = [];
    let capturedBytes = 0;
    let timedOut = false;
    let outputExceeded = false;
    let spawnError = null;
    let forceKillTimer = null;

    const capture = (target, chunk) => {
      const remaining = Math.max(0, MAX_CAPTURE_BYTES - capturedBytes);
      if (remaining > 0) target.push(chunk.subarray(0, remaining));
      capturedBytes += chunk.length;
      if (capturedBytes > MAX_CAPTURE_BYTES && !outputExceeded) {
        outputExceeded = true;
        terminateProcessTree(child, "SIGTERM");
        forceKillTimer ??= setTimeout(() => terminateProcessTree(child, "SIGKILL"), 1000);
      }
    };

    child.stdout.on("data", (chunk) => capture(stdoutChunks, chunk));
    child.stderr.on("data", (chunk) => capture(stderrChunks, chunk));
    child.on("error", (error) => {
      spawnError = error;
    });

    const timeoutTimer = setTimeout(() => {
      timedOut = true;
      terminateProcessTree(child, "SIGTERM");
      forceKillTimer ??= setTimeout(() => terminateProcessTree(child, "SIGKILL"), 1000);
    }, timeoutMs);

    child.on("close", (code, signal) => {
      clearTimeout(timeoutTimer);
      // Keep the delayed group SIGKILL after timeout/output overflow even if
      // the direct child exits first; a descendant may still ignore SIGTERM.
      let stderr = Buffer.concat(stderrChunks).toString("utf8");
      if (spawnError) stderr += `${stderr ? "\n" : ""}${spawnError.message}`;
      if (timedOut) stderr += `${stderr ? "\n" : ""}Timed out after ${Math.ceil(timeoutMs / 1000)} seconds.`;
      if (outputExceeded) stderr += `${stderr ? "\n" : ""}Captured output exceeded ${MAX_CAPTURE_BYTES} bytes.`;
      resolve({
        stdout: Buffer.concat(stdoutChunks).toString("utf8"),
        stderr,
        exitCode: timedOut ? 124 : outputExceeded ? 1 : Number.isInteger(code) ? code : 1,
        signal,
        timedOut,
        outputExceeded,
      });
    });
  });
}

function parseInput(argv) {
  const separator = argv.indexOf("--");
  const optionTokens = separator === -1 ? argv : argv.slice(0, separator);
  const requestedCommand = separator === -1 ? null : argv.slice(separator + 1);
  if (separator !== -1 && requestedCommand.length === 0) throw new Error("provide a command after -- or omit -- to use the criterion command");

  const options = {};
  for (let index = 0; index < optionTokens.length; index += 2) {
    const key = optionTokens[index];
    const value = optionTokens[index + 1];
    if (!key?.startsWith("--") || value === undefined) throw new Error("options must use --key value pairs before --");
    options[key.slice(2)] = value;
  }
  return { options, requestedCommand };
}

function requireString(options, key) {
  const value = options[key];
  if (typeof value !== "string" || value.trim() === "") throw new Error(`--${key} is required`);
  return value;
}

function parseExecutionProvenance(encoded) {
  if (encoded === undefined) return null;
  if (encoded.length > 32_768) throw new Error("--provenance is too large");
  try {
    const value = JSON.parse(Buffer.from(encoded, "base64url").toString("utf8"));
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("must decode to an object");
    return value;
  } catch (error) {
    throw new Error(`--provenance must be base64url-encoded JSON (${error.message})`);
  }
}

try {
  const { options, requestedCommand } = parseInput(process.argv.slice(2));
  const projectRoot = process.cwd();
  const taskPath = path.resolve(projectRoot, requireString(options, "task"));
  const projectRealPath = await realpath(projectRoot);
  const taskRealPath = await realpath(taskPath);
  if (!taskRealPath.startsWith(`${projectRealPath}${path.sep}`)) throw new Error("task file must be inside the project");
  const criterionId = requireString(options, "criterion");
  const { data: task } = await loadTask(taskRealPath);
  const taskErrors = validateTaskData(task);
  if (taskErrors.length > 0) throw new Error(`task contract is invalid: ${taskErrors.join("; ")}`);
  const criterion = task.acceptance_criteria.find((item) => item.id === criterionId);
  if (!criterion) throw new Error(`${criterionId} is not defined by ${task.id}`);
  if (criterion.verification_type !== "automated") throw new Error(`${criterionId} is not automated; use npm run agent:evidence with an actual source`);
  if (requestedCommand && !commandsEqual(requestedCommand, criterion.command)) {
    throw new Error("command after -- must exactly match the criterion command argv");
  }
  const command = criterion.command;
  const frozenCandidate = await currentCandidate(projectRoot);
  const frozenCriterionDigest = acceptanceCriterionDigest(criterion);
  const frozenTaskRevision = task.revision;
  const frozenTaskContractDigest = taskContractDigest(task);
  if (task.candidate_revision && task.candidate_revision !== frozenCandidate) {
    throw new Error(`candidate ${frozenCandidate} does not match task candidate_revision ${task.candidate_revision}`);
  }

  const taskId = safePathSegment(task.id, "task ID");
  const runId = safePathSegment(options["run-id"] ?? `RUN-${new Date().toISOString().replace(/[-:.TZ]/g, "")}-${randomUUID().slice(0, 8)}`, "run ID");
  const runRealPath = await ensureProjectDirectory(projectRoot, ["runs", taskId, runId]);
  const executionProvenance = parseExecutionProvenance(options.provenance);

  const result = await executeCommand(command, projectRoot, criterion.timeout_seconds * 1000);
  const stdout = result.stdout;
  const stderr = result.stderr;
  if (stdout) process.stdout.write(stdout);
  if (stderr) process.stderr.write(stderr);

  const exitCode = result.exitCode;
  const action = command.map((part) => JSON.stringify(part)).join(" ");
  const logPath = path.join(runRealPath, `command-${randomUUID().slice(0, 8)}.txt`);
  await writeFile(logPath, `COMMAND ${action}\nEXIT_CODE ${exitCode}\nTIMEOUT_SECONDS ${criterion.timeout_seconds}\nTIMED_OUT ${result.timedOut}\nOUTPUT_EXCEEDED ${result.outputExceeded}\nSIGNAL ${result.signal ?? "none"}\n\nSTDOUT\n${stdout}\nSTDERR\n${stderr}\n`, "utf8");

  const { data: currentTask } = await loadTask(taskRealPath);
  const currentTaskErrors = validateTaskData(currentTask);
  if (currentTaskErrors.length > 0) throw new Error(`task contract changed to an invalid state: ${currentTaskErrors.join("; ")}`);
  const currentCriterion = currentTask.acceptance_criteria.find((item) => item.id === criterionId);
  const observedCandidate = await currentCandidate(projectRoot);
  if (observedCandidate !== frozenCandidate) throw new Error(`candidate changed while check was running: ${frozenCandidate} -> ${observedCandidate}`);
  if (currentTask.revision !== frozenTaskRevision) throw new Error(`task revision changed while check was running: ${frozenTaskRevision} -> ${currentTask.revision}`);
  if (!currentCriterion || acceptanceCriterionDigest(currentCriterion) !== frozenCriterionDigest) {
    throw new Error("acceptance criterion changed while check was running");
  }
  if (taskContractDigest(currentTask) !== frozenTaskContractDigest) throw new Error("task acceptance contract changed while check was running");

  const { target } = await writeEvidenceRecord({
    projectRoot,
    task,
    criterion,
    runId,
    candidate: frozenCandidate,
    status: exitCode === 0 ? "passed" : "failed",
    environment: options.environment ?? "local",
    action,
    argv: command,
    exitCode,
    runner: "agent-check-v1",
    executionProvenance,
    summary: options.summary ?? (result.timedOut ? `Command timed out after ${criterion.timeout_seconds} seconds` : `Command exited with ${exitCode}`),
    artifacts: [path.relative(projectRoot, logPath)],
  });
  console.log(path.relative(projectRoot, target));
  console.log(`candidate_revision=${frozenCandidate}`);

  process.exitCode = exitCode;
} catch (error) {
  console.error(`Unable to run check: ${error.message}`);
  process.exitCode = 1;
}
