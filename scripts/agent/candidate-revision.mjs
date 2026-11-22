import { createHash } from "node:crypto";
import { lstat, readFile, readdir, readlink } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

// This project is currently a plain directory rather than a Git checkout, so
// it needs a deterministic filesystem provider. Keep the Git provider too: if
// the project is later put under Git, the provider change intentionally makes
// old local evidence stale instead of silently equating different snapshots.
const EXCLUDED_ROOT_ENTRIES = new Set([
  ".DS_Store",
  ".env",
  ".git",
  ".playwright-mcp",
  ".venv",
  "runs",
  "tasks",
  "venv",
]);

const EXCLUDED_DIRECTORY_NAMES = new Set(["__pycache__", ".pytest_cache", ".venv", "venv"]);
const EXCLUDED_PROJECT_DIRECTORIES = new Set(["frontend/dist", "frontend/node_modules", "htmlcov"]);
const EXCLUDED_PROJECT_FILES = new Set([
  ".claude/settings.local.json",
  ".coverage",
  "backend/run.log",
  "frontend/run.log",
  "语音链路对接参数-完整配置.md",
]);

function toPosix(relativePath) {
  return relativePath.split(path.sep).join("/");
}

export function isCandidatePathExcluded(relativePath) {
  const normalized = toPosix(relativePath).replace(/^\.\//, "");
  const parts = normalized.split("/");
  if (EXCLUDED_ROOT_ENTRIES.has(parts[0])) return true;
  if (parts.length === 1 && parts[0].startsWith(".env.") && parts[0] !== ".env.example") return true;
  if (parts.includes(".DS_Store")) return true;
  if (parts.some((part) => EXCLUDED_DIRECTORY_NAMES.has(part))) return true;
  if (normalized.endsWith(".pyc") || normalized.endsWith(".pyo")) return true;
  if (EXCLUDED_PROJECT_FILES.has(normalized)) return true;
  for (const excluded of EXCLUDED_PROJECT_DIRECTORIES) {
    if (normalized === excluded || normalized.startsWith(`${excluded}/`)) return true;
  }
  return false;
}

async function collectCandidateEntries(projectRoot, relativeDirectory = "") {
  const directory = path.join(projectRoot, relativeDirectory);
  const entries = await readdir(directory, { withFileTypes: true });
  entries.sort((left, right) => Buffer.from(left.name).compare(Buffer.from(right.name)));

  const snapshot = [];
  for (const entry of entries) {
    const relative = relativeDirectory ? path.join(relativeDirectory, entry.name) : entry.name;
    const normalized = toPosix(relative);
    if (isCandidatePathExcluded(normalized)) continue;

    const target = path.join(projectRoot, relative);
    const stat = await lstat(target);
    if (stat.isSymbolicLink()) {
      throw new Error(`symbolic links are not allowed in filesystem candidates: ${normalized}`);
    } else if (stat.isDirectory()) {
      snapshot.push(...await collectCandidateEntries(projectRoot, relative));
    } else if (stat.isFile()) {
      const contents = await readFile(target);
      const type = (stat.mode & 0o111) !== 0 ? "executable" : "file";
      snapshot.push([normalized, type, createHash("sha256").update(contents).digest("hex")]);
    } else {
      throw new Error(`unsupported candidate entry: ${normalized}`);
    }
  }
  return snapshot;
}

async function filesystemCandidate(projectRoot) {
  const snapshot = await collectCandidateEntries(path.resolve(projectRoot));
  snapshot.sort((left, right) => Buffer.from(left[0]).compare(Buffer.from(right[0])));
  const manifest = JSON.stringify({ schema: 3, provider: "filesystem", snapshot });
  return `snapshot-fs-v1-${createHash("sha256").update(manifest).digest("hex")}`;
}

function git(projectRoot, args) {
  const result = spawnSync("git", args, { cwd: projectRoot, encoding: "utf8" });
  if (result.status !== 0) throw new Error(result.stderr.trim() || `git ${args.join(" ")} failed`);
  return result.stdout;
}

function isGitWorktree(projectRoot) {
  const result = spawnSync("git", ["rev-parse", "--is-inside-work-tree"], { cwd: projectRoot, encoding: "utf8" });
  if (result.status === 0) return result.stdout.trim() === "true";
  if (result.status === 128 && /not a git repository/i.test(result.stderr)) return false;
  throw new Error(result.stderr.trim() || "unable to determine whether the project is a Git worktree");
}

function gitConfigBoolean(projectRoot, key, fallback = false) {
  const result = spawnSync("git", ["config", "--bool", "--get", key], { cwd: projectRoot, encoding: "utf8" });
  if (result.status === 1) return fallback;
  if (result.status !== 0) throw new Error(result.stderr.trim() || `git config ${key} failed`);
  return result.stdout.trim() === "true";
}

async function gitCandidate(projectRoot) {
  const productPathspec = ["--", ".", ":(exclude)tasks", ":(exclude)tasks/**", ":(exclude)runs", ":(exclude)runs/**"];
  const candidateFiles = git(projectRoot, ["ls-files", "--cached", "--others", "--exclude-standard", "-z", ...productPathspec])
    .split("\0").filter(Boolean).sort();
  const trackedModes = new Map(
    git(projectRoot, ["ls-files", "--stage", "-z", ...productPathspec])
      .split("\0").filter(Boolean).flatMap((entry) => {
        const separator = entry.indexOf("\t");
        if (separator === -1) return [];
        const [mode, , stage] = entry.slice(0, separator).split(" ");
        return stage === "0" ? [[entry.slice(separator + 1), mode]] : [];
      }),
  );
  const tracksExecutableMode = gitConfigBoolean(projectRoot, "core.filemode");
  const snapshot = [];

  for (const relative of candidateFiles) {
    const target = path.join(projectRoot, relative);
    let stat;
    try {
      stat = await lstat(target);
    } catch (error) {
      if (error.code === "ENOENT") continue;
      throw error;
    }
    const indexMode = trackedModes.get(relative);
    if (indexMode === "120000" || stat.isSymbolicLink()) {
      const linkTarget = stat.isSymbolicLink() ? await readlink(target, { encoding: "buffer" }) : await readFile(target);
      snapshot.push([relative, "symlink", createHash("sha256").update(linkTarget).digest("hex")]);
    } else if (stat.isFile()) {
      const executable = tracksExecutableMode ? (stat.mode & 0o111) !== 0 : indexMode === "100755";
      const filteredObjectId = git(projectRoot, ["hash-object", `--path=${relative}`, "--", relative]).trim();
      if (!/^[a-f0-9]{40,64}$/.test(filteredObjectId)) throw new Error(`unable to hash candidate file: ${relative}`);
      snapshot.push([relative, executable ? "executable" : "file", filteredObjectId]);
    } else {
      throw new Error(`unsupported candidate entry (use a project-specific snapshot adapter for submodules): ${relative}`);
    }
  }
  const manifest = JSON.stringify({ schema: 2, provider: "git", snapshot });
  return `snapshot-git-v2-${createHash("sha256").update(manifest).digest("hex")}`;
}

export async function currentCandidate(projectRoot) {
  const root = path.resolve(projectRoot);
  return isGitWorktree(root) ? gitCandidate(root) : filesystemCandidate(root);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    console.log(await currentCandidate(process.cwd()));
  } catch (error) {
    console.error(`Unable to observe candidate: ${error.message}`);
    process.exitCode = 1;
  }
}
