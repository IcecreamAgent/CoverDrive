import { createHash } from "node:crypto";
import { lstat, readdir, readFile, realpath } from "node:fs/promises";
import path from "node:path";

export const TASK_STATES = new Set([
  "proposed",
  "ready",
  "in_progress",
  "in_validation",
  "accepted",
  "released",
  "blocked",
  "needs_decision",
]);

export const EVIDENCE_STATES = new Set([
  "passed",
  "failed",
  "blocked",
  "not_run",
  "skipped",
]);

const RISK_LEVELS = new Set(["L0", "L1", "L2", "L3"]);
const AUTONOMY_LEVELS = new Set(["analysis", "implementation", "delivery"]);
const VERIFICATION_TYPES = new Set(["automated", "human_review", "user_study"]);
const DONE_STATES = new Set(["accepted", "released"]);

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

export function commandsEqual(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((part, index) => part === right[index]);
}

function isCommand(value) {
  return Array.isArray(value) && value.length > 0 && value.every(isNonEmptyString);
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (!isObject(value)) return value;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]));
}

export function parseTaskDocument(text, source = "task document") {
  const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/);
  if (!match) {
    throw new Error(`${source}: missing JSON-compatible YAML front matter`);
  }

  let data;
  try {
    data = JSON.parse(match[1]);
  } catch (error) {
    throw new Error(`${source}: front matter must be valid JSON-compatible YAML (${error.message})`);
  }

  if (!isObject(data)) {
    throw new Error(`${source}: front matter must contain an object`);
  }

  return { data, body: text.slice(match[0].length) };
}

export async function loadTask(filePath) {
  const text = await readFile(filePath, "utf8");
  return { ...parseTaskDocument(text, filePath), filePath };
}

export function acceptanceCriterionDigest(criterion) {
  const canonical = JSON.stringify({
    id: criterion.id,
    version: criterion.version,
    expected: criterion.expected,
    verification_type: criterion.verification_type,
    verification: criterion.verification,
    command: criterion.command ?? null,
    timeout_seconds: criterion.timeout_seconds ?? null,
    required: criterion.required,
  });
  return createHash("sha256").update(canonical).digest("hex");
}

export function taskContractDigest(task) {
  const contract = {
    schema_version: task.schema_version,
    id: task.id,
    revision: task.revision,
    goal: task.goal,
    user_scenario: task.user_scenario,
    scope: task.scope,
    non_scope: task.non_scope,
    done_at: task.done_at,
    acceptance_criteria: task.acceptance_criteria,
    risk_level: task.risk_level,
    risk_tags: task.risk_tags,
    autonomy: task.autonomy,
    context_refs: task.context_refs,
    baseline: task.baseline,
    dependencies: task.dependencies,
    write_boundary: task.write_boundary,
    execution_environment: task.execution_environment,
    human_gates: task.human_gates,
    required_evidence: task.required_evidence,
  };
  return createHash("sha256").update(JSON.stringify(canonicalize(contract))).digest("hex");
}

export function validateTaskData(data) {
  const errors = [];
  const requiredStrings = ["id", "goal", "user_scenario", "done_at", "risk_level", "autonomy", "owner", "next_action", "status"];

  if (data.schema_version !== 2) errors.push("schema_version must equal 2");
  if (!Number.isInteger(data.revision) || data.revision < 1) errors.push("revision must be an integer >= 1");

  for (const field of requiredStrings) {
    if (!isNonEmptyString(data[field])) errors.push(`${field} must be a non-empty string`);
  }

  if (!/^TASK-[A-Z0-9][A-Z0-9-]*$/.test(data.id ?? "")) errors.push("id must match TASK-...");
  if (!DONE_STATES.has(data.done_at)) errors.push("done_at must be accepted or released");
  if (!RISK_LEVELS.has(data.risk_level)) errors.push("risk_level must be L0, L1, L2, or L3");
  if (!AUTONOMY_LEVELS.has(data.autonomy)) errors.push("autonomy must be analysis, implementation, or delivery");
  if (!TASK_STATES.has(data.status)) errors.push(`status is not supported: ${data.status}`);

  for (const field of ["scope", "non_scope", "risk_tags", "context_refs", "dependencies", "write_boundary", "human_gates", "required_evidence"]) {
    if (!Array.isArray(data[field])) errors.push(`${field} must be an array`);
  }

  if (!isObject(data.execution_environment)) errors.push("execution_environment must be an object");
  if (!Number.isInteger(data.attempt) || data.attempt < 0) errors.push("attempt must be an integer >= 0");
  if (!isObject(data.budget)) {
    errors.push("budget must be an object");
  } else {
    const parallel = data.budget.max_parallel_subagents;
    const retries = data.budget.same_failure_retries_before_replan;
    if (!Number.isInteger(parallel) || parallel < 0 || parallel > 3) errors.push("budget.max_parallel_subagents must be an integer from 0 to 3");
    if (!Number.isInteger(retries) || retries < 0) errors.push("budget.same_failure_retries_before_replan must be an integer >= 0");
    if (data.budget.time_limit_minutes !== null && (!Number.isInteger(data.budget.time_limit_minutes) || data.budget.time_limit_minutes < 1)) {
      errors.push("budget.time_limit_minutes must be null or an integer >= 1");
    }
  }

  if (!Array.isArray(data.acceptance_criteria) || data.acceptance_criteria.length === 0) {
    errors.push("acceptance_criteria must contain at least one criterion");
  } else {
    const ids = new Set();
    for (const [index, criterion] of data.acceptance_criteria.entries()) {
      const prefix = `acceptance_criteria[${index}]`;
      if (!isObject(criterion)) {
        errors.push(`${prefix} must be an object`);
        continue;
      }
      if (!/^AC-[A-Z0-9][A-Z0-9-]*$/.test(criterion.id ?? "")) errors.push(`${prefix}.id must match AC-...`);
      if (ids.has(criterion.id)) errors.push(`${prefix}.id is duplicated`);
      ids.add(criterion.id);
      if (!Number.isInteger(criterion.version) || criterion.version < 1) errors.push(`${prefix}.version must be an integer >= 1`);
      if (!isNonEmptyString(criterion.expected)) errors.push(`${prefix}.expected must be a non-empty string`);
      if (!VERIFICATION_TYPES.has(criterion.verification_type)) errors.push(`${prefix}.verification_type is not supported`);
      if (!isNonEmptyString(criterion.verification)) errors.push(`${prefix}.verification must be a non-empty string`);
      if (criterion.verification_type === "automated" && !isCommand(criterion.command)) {
        errors.push(`${prefix}.command must be a non-empty argv array for automated verification`);
      }
      if (criterion.verification_type === "automated" && (!Number.isInteger(criterion.timeout_seconds) || criterion.timeout_seconds < 1 || criterion.timeout_seconds > 3600)) {
        errors.push(`${prefix}.timeout_seconds must be an integer from 1 to 3600 for automated verification`);
      }
      if (criterion.verification_type !== "automated" && criterion.command != null) {
        errors.push(`${prefix}.command must be null or omitted for non-automated verification`);
      }
      if (criterion.verification_type !== "automated" && criterion.timeout_seconds != null) {
        errors.push(`${prefix}.timeout_seconds must be null or omitted for non-automated verification`);
      }
      if (typeof criterion.required !== "boolean") errors.push(`${prefix}.required must be boolean`);
    }

    if (Array.isArray(data.required_evidence)) {
      for (const criterion of data.acceptance_criteria.filter((item) => item?.required)) {
        if (!data.required_evidence.includes(criterion.id)) errors.push(`required_evidence must include ${criterion.id}`);
      }
    }
  }

  const baselineRequired = new Set(["ready", "in_progress", "in_validation", "accepted", "released"]);
  if (baselineRequired.has(data.status) && !isNonEmptyString(data.baseline)) errors.push(`baseline is required when status is ${data.status}`);

  const candidateRequired = new Set(["in_validation", "accepted", "released"]);
  if (candidateRequired.has(data.status) && !isNonEmptyString(data.candidate_revision)) errors.push(`candidate_revision is required when status is ${data.status}`);

  const activeRunRequired = new Set(["in_progress", "in_validation"]);
  if (activeRunRequired.has(data.status) && !isNonEmptyString(data.active_run)) errors.push(`active_run is required when status is ${data.status}`);

  if (data.status === "blocked" && !isNonEmptyString(data.resume_condition)) errors.push("resume_condition is required when status is blocked");
  if (data.status === "released") {
    errors.push("released is disabled until a project-specific release adapter validates authorization, deployment, and post-release health evidence");
    if (data.done_at !== "released") errors.push("a released task must declare done_at as released");
    if (!isObject(data.release)) {
      errors.push("release must be an object when status is released");
    } else {
      if (!isNonEmptyString(data.release.target_environment)) errors.push("release.target_environment is required when status is released");
      if (!isNonEmptyString(data.release.authorization_ref)) errors.push("release.authorization_ref is required when status is released");
      if (data.release.deployed_revision !== data.candidate_revision) errors.push("release.deployed_revision must match candidate_revision");
      if (data.release.status !== "healthy") errors.push("release.status must be healthy before task status can be released");
      if (!Array.isArray(data.release.evidence) || data.release.evidence.length === 0) errors.push("release.evidence must contain post-release evidence");
    }
  }

  return errors;
}

export async function discoverTaskFiles(projectRoot) {
  const tasksDir = path.join(projectRoot, "tasks");
  let entries;
  try {
    const stat = await lstat(tasksDir);
    if (stat.isSymbolicLink()) throw new Error("tasks: symbolic links are not allowed for the active task root");
    if (!stat.isDirectory()) throw new Error("tasks: active task root must be a directory");
    const projectRealPath = await realpath(projectRoot);
    const tasksRealPath = await realpath(tasksDir);
    if (!tasksRealPath.startsWith(`${projectRealPath}${path.sep}`)) throw new Error("tasks: active task root escapes the project");
    entries = await readdir(tasksDir, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }

  const namedTaskEntries = entries.filter((entry) => /^TASK-[A-Z0-9][A-Z0-9-]*\.md$/.test(entry.name));
  const invalid = namedTaskEntries.find((entry) => entry.isSymbolicLink() || !entry.isFile());
  if (invalid) throw new Error(`tasks/${invalid.name}: active task entries must be regular files, not links or special files`);

  return namedTaskEntries
    .map((entry) => path.join(tasksDir, entry.name))
    .sort();
}

async function walkJsonFiles(directory, projectRoot) {
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return { files: [], errors: [] };
    throw error;
  }

  const files = [];
  const errors = [];
  for (const entry of entries) {
    const target = path.join(directory, entry.name);
    if (entry.isSymbolicLink()) {
      errors.push(`${path.relative(projectRoot, target)}: symbolic links are not allowed in evidence directories`);
    } else if (entry.isDirectory()) {
      const nested = await walkJsonFiles(target, projectRoot);
      files.push(...nested.files);
      errors.push(...nested.errors);
    }
    else if (entry.isFile() && entry.name.endsWith(".json")) files.push(target);
  }
  return { files, errors };
}

export async function loadTaskEvidence(projectRoot, taskId) {
  const evidence = [];
  const errors = [];
  const projectRealPath = await realpath(projectRoot);
  const runsPath = path.join(projectRoot, "runs");
  const taskRunsPath = path.join(projectRoot, "runs", taskId);
  let taskRunsRealPath;

  try {
    const runsStat = await lstat(runsPath);
    if (runsStat.isSymbolicLink()) return { evidence, errors: ["runs: symbolic links are not allowed for the evidence root"] };
    if (!runsStat.isDirectory()) return { evidence, errors: ["runs: evidence root must be a directory"] };
    const taskRunsStat = await lstat(taskRunsPath);
    if (taskRunsStat.isSymbolicLink()) return { evidence, errors: [`runs/${taskId}: symbolic links are not allowed for task evidence`] };
    if (!taskRunsStat.isDirectory()) return { evidence, errors: [`runs/${taskId}: task evidence path must be a directory`] };
    taskRunsRealPath = await realpath(taskRunsPath);
  } catch (error) {
    if (error.code === "ENOENT") return { evidence, errors };
    throw error;
  }
  if (!taskRunsRealPath.startsWith(`${projectRealPath}${path.sep}`)) {
    return { evidence, errors: [`runs/${taskId}: evidence directory escapes the project`] };
  }

  const walked = await walkJsonFiles(taskRunsRealPath, projectRealPath);
  errors.push(...walked.errors);

  for (const file of walked.files) {
    try {
      const record = JSON.parse(await readFile(file, "utf8"));
      evidence.push({ ...record, file });
    } catch (error) {
      errors.push(`${path.relative(projectRoot, file)}: invalid evidence JSON (${error.message})`);
    }
  }

  return { evidence, errors };
}

export function validateEvidenceRecord(record) {
  const errors = [];
  if (!isObject(record)) return ["evidence record must be an object"];
  if (record.schema_version !== 2) errors.push("schema_version must equal 2");
  for (const field of ["task_id", "task_contract_digest", "run_id", "criterion_id", "criterion_digest", "verification_type", "candidate_revision", "status", "environment", "action", "summary", "created_at"]) {
    if (!isNonEmptyString(record[field])) errors.push(`${field} must be a non-empty string`);
  }
  if (!Number.isInteger(record.task_revision) || record.task_revision < 1) errors.push("task_revision must be an integer >= 1");
  if (!Number.isInteger(record.criterion_version) || record.criterion_version < 1) errors.push("criterion_version must be an integer >= 1");
  if (!/^TASK-[A-Z0-9][A-Z0-9-]*$/.test(record.task_id ?? "")) errors.push("task_id must match TASK-...");
  if (!/^AC-[A-Z0-9][A-Z0-9-]*$/.test(record.criterion_id ?? "")) errors.push("criterion_id must match AC-...");
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(record.run_id ?? "") || record.run_id === "." || record.run_id === "..") errors.push("run_id contains unsupported characters");
  if (!/^[a-f0-9]{64}$/.test(record.task_contract_digest ?? "")) errors.push("task_contract_digest must be a SHA-256 digest");
  if (!/^[a-f0-9]{64}$/.test(record.criterion_digest ?? "")) errors.push("criterion_digest must be a SHA-256 digest");
  if (!VERIFICATION_TYPES.has(record.verification_type)) errors.push("verification_type is not supported");
  if (!EVIDENCE_STATES.has(record.status)) errors.push("status is not supported");
  if (!Array.isArray(record.artifacts)) errors.push("artifacts must be an array");
  if (!isObject(record.artifact_digests)) {
    errors.push("artifact_digests must be an object");
  } else if (Array.isArray(record.artifacts)) {
    for (const artifact of record.artifacts) {
      if (!/^[a-f0-9]{64}$/.test(record.artifact_digests[artifact] ?? "")) {
        errors.push(`artifact_digests must contain a SHA-256 digest for ${artifact}`);
      }
    }
    for (const artifact of Object.keys(record.artifact_digests)) {
      if (!record.artifacts.includes(artifact)) errors.push(`artifact_digests contains an unlisted artifact: ${artifact}`);
    }
  }
  if (Number.isNaN(Date.parse(record.created_at))) errors.push("created_at must be an ISO-compatible timestamp");
  if (record.status === "passed" && record.verification_type === "automated" && record.exit_code !== 0) errors.push("automated passed evidence requires exit_code 0");
  if (record.status === "passed" && record.verification_type === "automated" && record.runner !== "agent-check-v1") errors.push("automated passed evidence must be captured by agent-check-v1");
  if (record.verification_type === "automated" && !isCommand(record.argv)) errors.push("automated evidence requires a non-empty argv array");
  if (record.verification_type !== "automated" && record.argv != null) errors.push("non-automated evidence argv must be null");
  if (record.status === "passed" && record.verification_type === "automated" && record.artifacts?.length === 0) errors.push("automated passed evidence requires at least one captured artifact");
  if (record.status === "passed" && record.verification_type !== "automated" && !isNonEmptyString(record.source)) errors.push(`${record.verification_type} passed evidence requires source`);
  if (record.status === "skipped" && !isNonEmptyString(record.reason)) errors.push("skipped evidence requires reason");
  if (record.execution_provenance !== null && record.execution_provenance !== undefined && !isObject(record.execution_provenance)) {
    errors.push("execution_provenance must be null or an object");
  }
  if (record.environment === "local-ci-mode" && record.execution_provenance?.provider !== "local") {
    errors.push("local-ci-mode evidence requires local execution_provenance");
  }
  if (record.environment === "ci:github-actions") {
    const provenance = record.execution_provenance;
    if (!isObject(provenance) || provenance.provider !== "github-actions") {
      errors.push("ci:github-actions evidence requires GitHub Actions execution_provenance");
    } else {
      for (const field of ["run_id", "repository", "revision", "url"]) {
        if (!isNonEmptyString(provenance[field])) errors.push(`execution_provenance.${field} must be a non-empty string`);
      }
      if (!/^[a-f0-9]{40,64}$/.test(provenance.revision ?? "")) errors.push("execution_provenance.revision must be a Git object ID");
    }
  }
  return errors;
}

export async function validateEvidenceArtifacts(projectRoot, record, options = {}) {
  if (!Array.isArray(record.artifacts)) return [];
  const verifyDigests = options.verifyDigests !== false;
  const errors = [];
  const projectRealPath = await realpath(projectRoot);

  for (const [index, artifact] of record.artifacts.entries()) {
    const prefix = `artifacts[${index}]`;
    if (!isNonEmptyString(artifact)) {
      errors.push(`${prefix} must be a non-empty project-relative path`);
      continue;
    }
    if (path.isAbsolute(artifact)) {
      errors.push(`${prefix} must be project-relative`);
      continue;
    }

    const target = path.resolve(projectRealPath, artifact);
    if (!target.startsWith(`${projectRealPath}${path.sep}`)) {
      errors.push(`${prefix} escapes the project`);
      continue;
    }

    try {
      const targetRealPath = await realpath(target);
      if (!targetRealPath.startsWith(`${projectRealPath}${path.sep}`)) {
        errors.push(`${prefix} resolves outside the project`);
        continue;
      }
      if (record.verification_type === "automated") {
        const expectedArtifactRoot = path.join(projectRealPath, "runs", record.task_id, record.run_id);
        if (!targetRealPath.startsWith(`${expectedArtifactRoot}${path.sep}`)) {
          errors.push(`${prefix} for automated evidence must be inside runs/${record.task_id}/${record.run_id}`);
          continue;
        }
      }
      const stat = await lstat(targetRealPath);
      if (!stat.isFile()) errors.push(`${prefix} must reference an existing file`);
      if (stat.isFile() && verifyDigests) {
        const observedDigest = createHash("sha256").update(await readFile(targetRealPath)).digest("hex");
        if (record.artifact_digests?.[artifact] !== observedDigest) {
          errors.push(`${prefix} digest does not match the recorded artifact`);
        }
      }
    } catch (error) {
      if (error.code === "ENOENT") errors.push(`${prefix} does not exist: ${artifact}`);
      else throw error;
    }
  }

  return errors;
}

export function validateAcceptanceEvidence(task, evidence, options = {}) {
  if (!new Set(["accepted", "released"]).has(task.status)) return [];

  const errors = [];
  for (const criterion of task.acceptance_criteria.filter((item) => item.required)) {
    if (criterion.verification_type !== "automated") {
      errors.push(`${criterion.verification_type} criterion ${criterion.id} requires a project-specific trusted-source adapter before it can satisfy accepted`);
      continue;
    }
    const digest = acceptanceCriterionDigest(criterion);
    const matching = evidence
      .filter((record) =>
        record.task_id === task.id
        && record.task_revision === task.revision
        && record.task_contract_digest === taskContractDigest(task)
        && record.criterion_id === criterion.id
        && record.criterion_version === criterion.version
        && record.criterion_digest === digest
        && record.verification_type === criterion.verification_type
        && record.candidate_revision === task.candidate_revision
        && (criterion.verification_type !== "automated" || commandsEqual(record.argv, criterion.command))
        && (criterion.verification_type !== "automated" || !options.requiredAutomatedRun || record.run_id === options.requiredAutomatedRun)
      )
      .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at));
    const newestTime = matching.length > 0 ? Date.parse(matching[0].created_at) : null;
    const latest = newestTime === null ? [] : matching.filter((record) => Date.parse(record.created_at) === newestTime);
    if (latest.length === 0 || latest.some((record) => record.status !== "passed")) {
      const requiredRun = criterion.verification_type === "automated" ? options.requiredAutomatedRun : null;
      const runSuffix = requiredRun ? ` from automated run ${requiredRun}` : "";
      errors.push(`latest current evidence is not passed for ${criterion.id} v${criterion.version} on ${task.candidate_revision}${runSuffix}`);
    }
  }
  return errors;
}
