import assert from "node:assert/strict";
import { cp, mkdtemp, mkdir, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { currentCandidate, isCandidatePathExcluded } from "./candidate-revision.mjs";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const runCheck = path.join(scriptDirectory, "run-check.mjs");
const validateTask = path.join(scriptDirectory, "validate-task.mjs");
const ciVerify = path.join(scriptDirectory, "ci-verify.mjs");
const candidateRevision = path.join(scriptDirectory, "candidate-revision.mjs");

function execute(command, args, cwd, options = {}) {
  return spawnSync(command, args, { cwd, encoding: "utf8", ...options });
}

function taskDocument(command, overrides = {}, criterionOverrides = {}) {
  const data = {
    schema_version: 2,
    id: "TASK-CLI",
    revision: 1,
    goal: "Exercise the command contract",
    user_scenario: "A maintainer captures trustworthy evidence",
    scope: ["contract"],
    non_scope: [],
    done_at: "accepted",
    acceptance_criteria: [{
      id: "AC-01",
      version: 1,
      expected: "The declared command succeeds",
      verification_type: "automated",
      verification: "Run the declared command",
      command,
      timeout_seconds: 30,
      required: true,
      ...criterionOverrides,
    }],
    risk_level: "L1",
    risk_tags: [],
    autonomy: "implementation",
    context_refs: [],
    baseline: "baseline",
    dependencies: [],
    owner: "main-agent",
    write_boundary: [],
    execution_environment: {},
    human_gates: [],
    required_evidence: ["AC-01"],
    budget: { max_parallel_subagents: 1, same_failure_retries_before_replan: 1, time_limit_minutes: null },
    active_run: null,
    attempt: 0,
    candidate_revision: null,
    release: { target_environment: null, authorization_ref: null, deployed_revision: null, status: null, evidence: [] },
    next_action: "Run the check",
    resume_condition: null,
    status: "ready",
    ...overrides,
  };
  return `---\n${JSON.stringify(data, null, 2)}\n---\n\n# Test task\n`;
}

async function createRepository(command, overrides, criterionOverrides) {
  const root = await mkdtemp(path.join(tmpdir(), "agent-contract-"));
  await mkdir(path.join(root, "tasks"));
  await writeFile(path.join(root, "product.txt"), "original\n", "utf8");
  await writeFile(path.join(root, "tasks/TASK-CLI.md"), taskDocument(command, overrides, criterionOverrides), "utf8");
  assert.equal(execute("git", ["init", "-q"], root).status, 0);
  assert.equal(execute("git", ["add", "."], root).status, 0);
  assert.equal(execute("git", ["-c", "user.name=Contract Test", "-c", "user.email=contract@example.test", "commit", "-qm", "fixture"], root).status, 0);
  return root;
}

async function jsonEvidence(root) {
  const taskRoot = path.join(root, "runs/TASK-CLI");
  let runDirectories = [];
  try {
    runDirectories = await readdir(taskRoot);
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const records = [];
  for (const run of runDirectories) {
    for (const entry of await readdir(path.join(taskRoot, run))) {
      if (entry.endsWith(".json")) records.push(path.join(taskRoot, run, entry));
    }
  }
  return records;
}

test("rejects a substituted command before it executes", async (context) => {
  const declared = [process.execPath, "-e", "process.stdout.write('declared')"];
  const root = await createRepository(declared);
  context.after(() => rm(root, { recursive: true, force: true }));
  const sentinel = path.join(root, "wrong-command-ran");
  const substituted = [process.execPath, "-e", `require('node:fs').writeFileSync(${JSON.stringify(sentinel)}, 'bad')`];

  const result = execute(process.execPath, [runCheck, "--task", "tasks/TASK-CLI.md", "--criterion", "AC-01", "--", ...substituted], root);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /must exactly match/);
  await assert.rejects(readFile(sentinel), { code: "ENOENT" });
  assert.deepEqual(await jsonEvidence(root), []);
});

test("executes the declared argv and captures a project-local text artifact", async (context) => {
  const declared = [process.execPath, "-e", "process.stdout.write('declared ok')"];
  const root = await createRepository(declared);
  context.after(() => rm(root, { recursive: true, force: true }));

  const result = execute(process.execPath, [runCheck, "--task", "tasks/TASK-CLI.md", "--criterion", "AC-01"], root);
  assert.equal(result.status, 0, result.stderr);
  const records = await jsonEvidence(root);
  assert.equal(records.length, 1);
  const record = JSON.parse(await readFile(records[0], "utf8"));
  assert.deepEqual(record.argv, declared);
  assert.equal(record.status, "passed");
  assert.equal(record.exit_code, 0);
  assert.equal(record.artifacts.length, 1);
  assert.match(record.artifacts[0], /\.txt$/);
  assert.match(await readFile(path.join(root, record.artifacts[0]), "utf8"), /declared ok/);
  assert.match(record.artifact_digests[record.artifacts[0]], /^[a-f0-9]{64}$/);

  await writeFile(path.join(root, record.artifacts[0]), "tampered\n", "utf8");
  const tampered = execute(process.execPath, [validateTask, "tasks/TASK-CLI.md"], root);
  assert.notEqual(tampered.status, 0);
  assert.match(tampered.stderr, /digest does not match/);
});

test("does not record a pass when product code changes during the check", async (context) => {
  const mutation = "require('node:fs').writeFileSync('product.txt', 'changed\\n')";
  const root = await createRepository([process.execPath, "-e", mutation]);
  context.after(() => rm(root, { recursive: true, force: true }));

  const result = execute(process.execPath, [runCheck, "--task", "tasks/TASK-CLI.md", "--criterion", "AC-01"], root);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /candidate changed while check was running/);
  assert.deepEqual(await jsonEvidence(root), []);
});

test("does not record a pass when the criterion changes during the check", async (context) => {
  const mutation = "const fs=require('node:fs');const p='tasks/TASK-CLI.md';fs.writeFileSync(p,fs.readFileSync(p,'utf8').replace('The declared command succeeds','A changed expectation'))";
  const root = await createRepository([process.execPath, "-e", mutation]);
  context.after(() => rm(root, { recursive: true, force: true }));

  const result = execute(process.execPath, [runCheck, "--task", "tasks/TASK-CLI.md", "--criterion", "AC-01"], root);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /acceptance criterion changed while check was running/);
  assert.deepEqual(await jsonEvidence(root), []);
});

test("does not record a pass when another acceptance gate changes during the check", async (context) => {
  const mutation = "const fs=require('node:fs');const p='tasks/TASK-CLI.md';fs.writeFileSync(p,fs.readFileSync(p,'utf8').replace('\\\"human_gates\\\": []','\\\"human_gates\\\": [\\\"new gate\\\"]'))";
  const root = await createRepository([process.execPath, "-e", mutation]);
  context.after(() => rm(root, { recursive: true, force: true }));

  const result = execute(process.execPath, [runCheck, "--task", "tasks/TASK-CLI.md", "--criterion", "AC-01"], root);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /task acceptance contract changed while check was running/);
  assert.deepEqual(await jsonEvidence(root), []);
});

test("times out a hung command and records failed evidence", async (context) => {
  const root = await createRepository(
    [process.execPath, "-e", "setTimeout(() => {}, 5000)"],
    undefined,
    { timeout_seconds: 1 },
  );
  context.after(() => rm(root, { recursive: true, force: true }));
  const startedAt = Date.now();
  const result = execute(process.execPath, [runCheck, "--task", "tasks/TASK-CLI.md", "--criterion", "AC-01"], root);
  assert.notEqual(result.status, 0);
  assert.ok(Date.now() - startedAt < 4000, "command should be terminated near its criterion timeout");
  assert.match(result.stderr, /Timed out after 1 seconds/);
  const [recordPath] = await jsonEvidence(root);
  const record = JSON.parse(await readFile(recordPath, "utf8"));
  assert.equal(record.status, "failed");
  assert.equal(record.exit_code, 124);
  assert.match(await readFile(path.join(root, record.artifacts[0]), "utf8"), /TIMED_OUT true/);
});

test("force-kills a descendant that ignores the timeout SIGTERM", { skip: process.platform === "win32" }, async (context) => {
  const sentinel = path.join(tmpdir(), `agent-descendant-${Date.now()}-${Math.random()}`);
  const descendant = `process.on('SIGTERM',()=>{});setTimeout(()=>require('node:fs').writeFileSync(${JSON.stringify(sentinel)},'survived'),2600);setInterval(()=>{},1000)`;
  const parent = `const{spawn}=require('node:child_process');const child=spawn(process.execPath,['-e',${JSON.stringify(descendant)}],{stdio:'ignore'});child.unref();process.on('SIGTERM',()=>process.exit(0));setInterval(()=>{},1000)`;
  const root = await createRepository(
    [process.execPath, "-e", parent],
    undefined,
    { timeout_seconds: 1 },
  );
  context.after(() => Promise.all([
    rm(root, { recursive: true, force: true }),
    rm(sentinel, { force: true }),
  ]));

  const result = execute(process.execPath, [runCheck, "--task", "tasks/TASK-CLI.md", "--criterion", "AC-01"], root);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Timed out after 1 seconds/);
  await new Promise((resolve) => setTimeout(resolve, 1000));
  await assert.rejects(readFile(sentinel), { code: "ENOENT" });
});

test("rejects a symlinked runs root without writing outside the project", async (context) => {
  const sentinelCommand = [process.execPath, "-e", "require('node:fs').writeFileSync('command-ran', 'bad')"];
  const root = await createRepository(sentinelCommand);
  const outside = await mkdtemp(path.join(tmpdir(), "agent-contract-outside-"));
  context.after(() => Promise.all([
    rm(root, { recursive: true, force: true }),
    rm(outside, { recursive: true, force: true }),
  ]));
  await symlink(outside, path.join(root, "runs"));

  const result = execute(process.execPath, [runCheck, "--task", "tasks/TASK-CLI.md", "--criterion", "AC-01"], root);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /must not be a symbolic link/);
  assert.deepEqual(await readdir(outside), []);

  const validation = execute(process.execPath, [validateTask, "tasks/TASK-CLI.md"], root);
  assert.notEqual(validation.status, 0);
  assert.match(validation.stderr, /symbolic links are not allowed/);
});

test("rejects symlinked or external task files", async (context) => {
  const root = await createRepository([process.execPath, "-e", "process.exit(0)"]);
  const outside = await mkdtemp(path.join(tmpdir(), "agent-task-outside-"));
  context.after(() => Promise.all([
    rm(root, { recursive: true, force: true }),
    rm(outside, { recursive: true, force: true }),
  ]));
  const externalTask = path.join(outside, "TASK-CLI.md");
  await writeFile(externalTask, taskDocument([process.execPath, "-e", "process.exit(0)"]), "utf8");

  const explicit = execute(process.execPath, [validateTask, externalTask], root);
  assert.notEqual(explicit.status, 0);
  assert.match(explicit.stderr, /task file must be inside the project/);

  await symlink(externalTask, path.join(root, "tasks/TASK-LINK.md"));
  const linkedEntry = execute(process.execPath, [validateTask], root);
  assert.notEqual(linkedEntry.status, 0);
  assert.match(linkedEntry.stderr, /active task entries must be regular files/);

  await rm(path.join(root, "tasks"), { recursive: true });
  await symlink(outside, path.join(root, "tasks"));
  const discovered = execute(process.execPath, [validateTask], root);
  assert.notEqual(discovered.status, 0);
  assert.match(discovered.stderr, /symbolic links are not allowed for the active task root/);
});

test("final validation rejects evidence after the product candidate changes", async (context) => {
  const declared = [process.execPath, "-e", "process.stdout.write('stable')"];
  const root = await createRepository(declared);
  context.after(() => rm(root, { recursive: true, force: true }));

  const checked = execute(process.execPath, [runCheck, "--task", "tasks/TASK-CLI.md", "--criterion", "AC-01"], root);
  assert.equal(checked.status, 0, checked.stderr);
  const [recordPath] = await jsonEvidence(root);
  const record = JSON.parse(await readFile(recordPath, "utf8"));
  const taskPath = path.join(root, "tasks/TASK-CLI.md");
  const readyTask = await readFile(taskPath, "utf8");
  const acceptedTask = readyTask
    .replace('"candidate_revision": null', `"candidate_revision": ${JSON.stringify(record.candidate_revision)}`)
    .replace('"status": "ready"', '"status": "accepted"');
  await writeFile(taskPath, acceptedTask, "utf8");

  const accepted = execute(process.execPath, [validateTask], root);
  assert.equal(accepted.status, 0, accepted.stderr);
  await writeFile(path.join(root, "product.txt"), "changed after check\n", "utf8");
  const stale = execute(process.execPath, [validateTask], root);
  assert.notEqual(stale.status, 0);
  assert.match(stale.stderr, /does not match current project candidate/);
});

test("candidate manifests distinguish path-content boundary collisions", async (context) => {
  const root = await createRepository([process.execPath, "-e", "process.exit(0)"]);
  context.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(path.join(root, "a"), "bc", "utf8");
  const first = await currentCandidate(root);
  await rm(path.join(root, "a"));
  await writeFile(path.join(root, "ab"), "c", "utf8");
  const second = await currentCandidate(root);
  assert.notEqual(first, second);
});

test("voice-agent filesystem snapshot excludes local state and generated outputs", async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), "voice-agent-snapshot-"));
  context.after(() => rm(root, { recursive: true, force: true }));
  for (const directory of [
    "frontend/src", "frontend/dist/assets", "frontend/node_modules/pkg",
    "backend/__pycache__", "backend/.pytest_cache", "backend/.venv/lib",
    ".claude", ".playwright-mcp", "htmlcov", "tasks", "runs/TASK-X/RUN-X",
  ]) await mkdir(path.join(root, directory), { recursive: true });
  await writeFile(path.join(root, "frontend/src/main.jsx"), "export const value = 1;\n");
  await writeFile(path.join(root, "frontend/package-lock.json"), "lock-v1\n");
  await writeFile(path.join(root, ".env.example"), "TOKEN=\n");
  const excludedFiles = [
    ".env", ".env.local", ".DS_Store", ".coverage", ".claude/settings.local.json",
    ".playwright-mcp/page.yml", "htmlcov/index.html", "语音链路对接参数-完整配置.md",
    "frontend/dist/assets/app.js", "frontend/node_modules/pkg/index.js", "frontend/run.log",
    "backend/__pycache__/app.pyc", "backend/.pytest_cache/state", "backend/.venv/lib/site.py",
    "backend/run.log", "tasks/TASK-X.md", "runs/TASK-X/RUN-X/evidence.json",
  ];
  for (const relative of excludedFiles) await writeFile(path.join(root, relative), `v1:${relative}\n`);

  const baseline = await currentCandidate(root);
  for (const relative of excludedFiles) await writeFile(path.join(root, relative), `v2:${relative}\n`);
  assert.equal(await currentCandidate(root), baseline);
});

test("voice-agent filesystem snapshot dynamically includes product and delivery inputs", async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), "voice-agent-included-"));
  context.after(() => rm(root, { recursive: true, force: true }));
  const includedFiles = [
    "backend/app/main.py",
    "backend/tests/test_voice_socket.py",
    "backend/models/silero_vad.onnx",
    "backend/requirements.txt",
    "frontend/src/main.jsx",
    "frontend/public/pcm-worklet.js",
    "frontend/package-lock.json",
    ".env.example",
    "package-lock.json",
    "PROJECT_PROFILE.md",
    "scripts/agent/candidate-revision.mjs",
  ];
  for (const relative of includedFiles) {
    await mkdir(path.dirname(path.join(root, relative)), { recursive: true });
    await writeFile(path.join(root, relative), Buffer.from(`v1:${relative}\n`));
  }
  const baseline = await currentCandidate(root);
  for (const relative of includedFiles) {
    const target = path.join(root, relative);
    const original = await readFile(target);
    await writeFile(target, Buffer.concat([original, Buffer.from([0, 255, 10])]));
    assert.notEqual(await currentCandidate(root), baseline, relative);
    await writeFile(target, original);
    assert.equal(await currentCandidate(root), baseline, `${relative} restore`);
  }
});

test("filesystem candidates reject source symlinks instead of missing target drift", async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), "voice-agent-link-"));
  const outside = await mkdtemp(path.join(tmpdir(), "voice-agent-link-target-"));
  context.after(() => Promise.all([
    rm(root, { recursive: true, force: true }),
    rm(outside, { recursive: true, force: true }),
  ]));
  await mkdir(path.join(root, "frontend/src"), { recursive: true });
  await writeFile(path.join(outside, "config.js"), "export const endpoint = 'one';\n");
  await symlink(path.join(outside, "config.js"), path.join(root, "frontend/src/config.js"));
  await assert.rejects(currentCandidate(root), /symbolic links are not allowed/);
});

test("candidate revision has a direct CLI output", async (context) => {
  const root = await mkdtemp(path.join(tmpdir(), "voice-agent-candidate-cli-"));
  context.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(path.join(root, "product.txt"), "candidate\n");
  const result = execute(process.execPath, [candidateRevision], root);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout.trim(), /^snapshot-fs-v1-[a-f0-9]{64}$/);
});

test("voice-agent rules keep source, tests, config, models, and lockfiles in the candidate", () => {
  for (const candidatePath of [
    "backend/app/main.py",
    "backend/tests/test_voice_socket.py",
    "backend/models/silero_vad.onnx",
    "backend/requirements.txt",
    "frontend/src/main.jsx",
    "frontend/package-lock.json",
    ".env.example",
    "PROJECT_PROFILE.md",
    "scripts/agent/candidate-revision.mjs",
  ]) assert.equal(isCandidatePathExcluded(candidatePath), false, candidatePath);

  for (const generatedPath of [
    ".env",
    ".env.local",
    ".claude/settings.local.json",
    ".playwright-mcp/page.yml",
    ".venv/lib/site.py",
    "backend/.venv/lib/site.py",
    "venv/lib/site.py",
    ".coverage",
    "语音链路对接参数-完整配置.md",
    "htmlcov/index.html",
    "tasks/TASK-1.md",
    "runs/TASK-1/RUN-1/evidence.json",
    "frontend/dist/index.html",
    "frontend/node_modules/react/index.js",
    "backend/__pycache__/main.pyc",
    "backend/.pytest_cache/state",
    "backend/run.log",
    "frontend/run.log",
  ]) assert.equal(isCandidatePathExcluded(generatedPath), true, generatedPath);
});

test("candidate snapshot order is deterministic across repositories", async (context) => {
  const firstRoot = await createRepository([process.execPath, "-e", "process.exit(0)"]);
  const secondRoot = await createRepository([process.execPath, "-e", "process.exit(0)"]);
  context.after(() => Promise.all([
    rm(firstRoot, { recursive: true, force: true }),
    rm(secondRoot, { recursive: true, force: true }),
  ]));
  for (const [name, contents] of [["a", "lower"], ["Z", "upper"], ["ä", "accent"]]) {
    await writeFile(path.join(firstRoot, name), contents, "utf8");
  }
  for (const [name, contents] of [["ä", "accent"], ["Z", "upper"], ["a", "lower"]]) {
    await writeFile(path.join(secondRoot, name), contents, "utf8");
  }
  assert.equal(await currentCandidate(firstRoot), await currentCandidate(secondRoot));
});

test("candidate content uses Git clean filters for line endings", async (context) => {
  const root = await createRepository([process.execPath, "-e", "process.exit(0)"]);
  context.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(path.join(root, ".gitattributes"), "*.txt text eol=lf\n", "utf8");
  await writeFile(path.join(root, "product.txt"), "first\nsecond\n", "utf8");
  assert.equal(execute("git", ["add", ".gitattributes", "product.txt"], root).status, 0);
  assert.equal(execute("git", ["-c", "user.name=Contract Test", "-c", "user.email=contract@example.test", "commit", "-qm", "line ending policy"], root).status, 0);
  const lfCandidate = await currentCandidate(root);
  await writeFile(path.join(root, "product.txt"), "first\r\nsecond\r\n", "utf8");
  assert.equal(await currentCandidate(root), lfCandidate);
});

test("tracked symlinks keep one candidate when core.symlinks materializes them as files", async (context) => {
  const root = await createRepository([process.execPath, "-e", "process.exit(0)"]);
  context.after(() => rm(root, { recursive: true, force: true }));
  const linkPath = path.join(root, "product-link");
  await symlink("product.txt", linkPath);
  assert.equal(execute("git", ["add", "product-link"], root).status, 0);
  assert.equal(execute("git", ["-c", "user.name=Contract Test", "-c", "user.email=contract@example.test", "commit", "-qm", "tracked link"], root).status, 0);
  const symlinkCandidate = await currentCandidate(root);

  await rm(linkPath);
  await writeFile(linkPath, "product.txt", "utf8");
  assert.equal(execute("git", ["config", "core.symlinks", "false"], root).status, 0);
  assert.equal(await currentCandidate(root), symlinkCandidate);
});

test("task and evidence-only commits do not change the product snapshot", async (context) => {
  const root = await createRepository([process.execPath, "-e", "process.exit(0)"]);
  context.after(() => rm(root, { recursive: true, force: true }));
  const before = await currentCandidate(root);
  await writeFile(path.join(root, "tasks/TASK-CLI.md"), `${await readFile(path.join(root, "tasks/TASK-CLI.md"), "utf8")}\nState note.\n`, "utf8");
  await mkdir(path.join(root, "runs/TASK-CLI/RUN-STATE"), { recursive: true });
  await writeFile(path.join(root, "runs/TASK-CLI/RUN-STATE/note.txt"), "state only\n", "utf8");
  assert.equal(execute("git", ["add", "tasks", "runs"], root).status, 0);
  assert.equal(execute("git", ["-c", "user.name=Contract Test", "-c", "user.email=contract@example.test", "commit", "-qm", "persist state"], root).status, 0);
  assert.equal(await currentCandidate(root), before);
});

test("CI replays accepted automated criteria after state evidence is committed", async (context) => {
  const declared = [process.execPath, "-e", "process.stdout.write('ci replay')"];
  const root = await createRepository(declared);
  context.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "scripts"));
  await cp(scriptDirectory, path.join(root, "scripts/agent"), { recursive: true });
  assert.equal(execute("git", ["add", "scripts"], root).status, 0);
  assert.equal(execute("git", ["-c", "user.name=Contract Test", "-c", "user.email=contract@example.test", "commit", "-qm", "agent runtime"], root).status, 0);

  const checked = execute(process.execPath, [runCheck, "--task", "tasks/TASK-CLI.md", "--criterion", "AC-01"], root);
  assert.equal(checked.status, 0, checked.stderr);
  const [recordPath] = await jsonEvidence(root);
  const record = JSON.parse(await readFile(recordPath, "utf8"));
  const taskPath = path.join(root, "tasks/TASK-CLI.md");
  const acceptedTask = (await readFile(taskPath, "utf8"))
    .replace('"candidate_revision": null', `"candidate_revision": ${JSON.stringify(record.candidate_revision)}`)
    .replace('"status": "ready"', '"status": "accepted"');
  await writeFile(taskPath, acceptedTask, "utf8");
  assert.equal(execute("git", ["add", "tasks", "runs"], root).status, 0);
  assert.equal(execute("git", ["-c", "user.name=Contract Test", "-c", "user.email=contract@example.test", "commit", "-qm", "accepted evidence"], root).status, 0);
  assert.equal(await currentCandidate(root), record.candidate_revision);

  const localReplay = execute(process.execPath, [ciVerify, "--require-replay"], root);
  assert.equal(localReplay.status, 0, localReplay.stderr);
  assert.match(localReplay.stdout, /Local CI-mode acceptance replay passed/);
  assert.match(localReplay.stdout, /does not satisfy the remote required CI check/);

  const githubEnvironment = {
    ...process.env,
    CI: "true",
    GITHUB_ACTIONS: "true",
    GITHUB_RUN_ID: "123456",
    GITHUB_RUN_ATTEMPT: "1",
    GITHUB_SHA: "a".repeat(40),
    GITHUB_REPOSITORY: "example/product",
    GITHUB_SERVER_URL: "https://github.example.test",
  };
  const ciReplay = execute(process.execPath, [ciVerify, "--require-ci", "--require-replay"], root, { env: githubEnvironment });
  assert.equal(ciReplay.status, 0, ciReplay.stderr);
  assert.match(ciReplay.stdout, /CI acceptance replay passed/);

  const replayRecords = await Promise.all((await jsonEvidence(root)).map(async (file) => JSON.parse(await readFile(file, "utf8"))));
  const localRecord = replayRecords.find((item) => item.run_id.startsWith("LOCAL-CI-"));
  assert.equal(localRecord.environment, "local-ci-mode");
  assert.equal(localRecord.execution_provenance.provider, "local");
  const ciRecord = replayRecords.find((item) => item.run_id.startsWith("CI-"));
  assert.equal(ciRecord.environment, "ci:github-actions");
  assert.equal(ciRecord.execution_provenance.run_id, "123456");
  assert.equal(ciRecord.execution_provenance.revision, "a".repeat(40));
});

test("required CI and required replay fail closed when their preconditions are absent", async (context) => {
  const root = await createRepository([process.execPath, "-e", "process.exit(0)"]);
  context.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "scripts"));
  await cp(scriptDirectory, path.join(root, "scripts/agent"), { recursive: true });

  const notCi = execute(process.execPath, [ciVerify, "--require-ci"], root, {
    env: { ...process.env, CI: "", GITHUB_ACTIONS: "", GITHUB_RUN_ID: "", GITHUB_SHA: "", GITHUB_REPOSITORY: "" },
  });
  assert.notEqual(notCi.status, 0);
  assert.match(notCi.stderr, /trusted CI environment is required/);

  const noReplay = execute(process.execPath, [ciVerify, "--require-replay"], root);
  assert.notEqual(noReplay.status, 0);
  assert.match(noReplay.stderr, /No accepted or released automated acceptance criteria/);
});
