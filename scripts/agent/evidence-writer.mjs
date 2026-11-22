import { createHash, randomUUID } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  acceptanceCriterionDigest,
  taskContractDigest,
  validateEvidenceArtifacts,
  validateEvidenceRecord,
} from "./task-contract.mjs";
import { ensureProjectDirectory, safePathSegment } from "./path-safety.mjs";

export async function writeEvidenceRecord({
  projectRoot,
  task,
  criterion,
  runId,
  candidate,
  status,
  environment,
  action,
  argv,
  exitCode,
  runner,
  executionProvenance = null,
  summary,
  source = null,
  reason = null,
  artifacts = [],
  createdAt = new Date().toISOString(),
}) {
  const taskId = safePathSegment(task.id, "task ID");
  const safeRunId = safePathSegment(runId, "run ID");
  const criterionId = safePathSegment(criterion.id, "criterion ID");
  const record = {
    schema_version: 2,
    task_id: task.id,
    task_revision: task.revision,
    task_contract_digest: taskContractDigest(task),
    run_id: safeRunId,
    criterion_id: criterion.id,
    criterion_version: criterion.version,
    criterion_digest: acceptanceCriterionDigest(criterion),
    verification_type: criterion.verification_type,
    candidate_revision: candidate,
    status,
    environment,
    action,
    argv,
    exit_code: exitCode,
    runner,
    execution_provenance: executionProvenance,
    summary,
    source,
    reason,
    artifacts,
    artifact_digests: {},
    created_at: createdAt,
  };

  const artifactErrors = await validateEvidenceArtifacts(projectRoot, record, { verifyDigests: false });
  if (artifactErrors.length > 0) throw new Error(`evidence artifacts are invalid: ${artifactErrors.join("; ")}`);
  for (const artifact of artifacts) {
    record.artifact_digests[artifact] = createHash("sha256")
      .update(await readFile(path.resolve(projectRoot, artifact)))
      .digest("hex");
  }

  const errors = validateEvidenceRecord(record);
  errors.push(...await validateEvidenceArtifacts(projectRoot, record));
  if (errors.length > 0) throw new Error(`evidence record is invalid: ${errors.join("; ")}`);

  const targetDirectory = await ensureProjectDirectory(projectRoot, ["runs", taskId, safeRunId]);
  const target = path.join(targetDirectory, `${criterionId}-${randomUUID().slice(0, 8)}.json`);
  await writeFile(target, `${JSON.stringify(record, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  return { record, target };
}
