#!/usr/bin/env node

import path from "node:path";
import process from "node:process";
import { lstat, realpath } from "node:fs/promises";
import {
  discoverTaskFiles,
  loadTask,
  loadTaskEvidence,
  validateAcceptanceEvidence,
  validateEvidenceArtifacts,
  validateEvidenceRecord,
  validateTaskData,
} from "./task-contract.mjs";
import { currentCandidate } from "./candidate-revision.mjs";

const projectRoot = process.cwd();
const projectRealPath = await realpath(projectRoot);
const argv = process.argv.slice(2);
let requiredAutomatedRun = null;
const explicitFileArgs = [];
for (let index = 0; index < argv.length; index += 1) {
  if (argv[index] === "--require-automated-run") {
    requiredAutomatedRun = argv[index + 1];
    if (!requiredAutomatedRun || requiredAutomatedRun.startsWith("--")) {
      throw new Error("--require-automated-run requires a run ID");
    }
    index += 1;
  } else {
    explicitFileArgs.push(argv[index]);
  }
}
const explicitFiles = explicitFileArgs.map((file) => path.resolve(projectRoot, file));
const files = explicitFiles.length > 0 ? explicitFiles : await discoverTaskFiles(projectRoot);
const activeTasksDirectory = path.join(projectRealPath, "tasks");
let observedCandidate;

if (files.length === 0) {
  console.log("OK: no active task files found.");
  process.exit(0);
}

let failed = false;

for (const file of files) {
  const relative = path.relative(projectRoot, file);
  try {
    const fileStat = await lstat(file);
    if (fileStat.isSymbolicLink()) throw new Error("task file must not be a symbolic link");
    if (!fileStat.isFile()) throw new Error("task path must be a file");
    const taskRealPath = await realpath(file);
    if (!taskRealPath.startsWith(`${projectRealPath}${path.sep}`)) throw new Error("task file must be inside the project");
    const { data } = await loadTask(taskRealPath);
    const errors = validateTaskData(data);

    if (path.dirname(taskRealPath) === activeTasksDirectory) {
      const expectedName = `${data.id}.md`;
      if (path.basename(taskRealPath) !== expectedName) errors.push(`filename must be ${expectedName}`);
      if (new Set(["in_validation", "accepted", "released"]).has(data.status) && data.candidate_revision) {
        observedCandidate ??= await currentCandidate(projectRoot);
        if (data.candidate_revision !== observedCandidate) {
          errors.push(`candidate_revision ${data.candidate_revision} does not match current project candidate ${observedCandidate}`);
        }
      }
    }

    const loaded = await loadTaskEvidence(projectRoot, data.id);
    errors.push(...loaded.errors);
    for (const record of loaded.evidence) {
      for (const error of validateEvidenceRecord(record)) {
        errors.push(`${path.relative(projectRoot, record.file)}: ${error}`);
      }
      for (const error of await validateEvidenceArtifacts(projectRoot, record)) {
        errors.push(`${path.relative(projectRoot, record.file)}: ${error}`);
      }
    }
    errors.push(...validateAcceptanceEvidence(data, loaded.evidence, { requiredAutomatedRun }));

    if (errors.length > 0) {
      failed = true;
      console.error(`FAIL: ${relative}`);
      for (const error of errors) console.error(`  - ${error}`);
    } else {
      console.log(`OK: ${relative}`);
    }
  } catch (error) {
    failed = true;
    console.error(`FAIL: ${relative}`);
    console.error(`  - ${error.message}`);
  }
}

process.exitCode = failed ? 1 : 0;
