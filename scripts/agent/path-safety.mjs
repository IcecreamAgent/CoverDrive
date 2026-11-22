import { lstat, mkdir, realpath } from "node:fs/promises";
import path from "node:path";

export function isPathInside(parent, child) {
  return child === parent || child.startsWith(`${parent}${path.sep}`);
}

export function safePathSegment(value, label) {
  if (value === "." || value === ".." || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value)) {
    throw new Error(`${label} contains unsupported characters`);
  }
  return value;
}

async function lstatIfPresent(target) {
  try {
    return await lstat(target);
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

export async function ensureProjectDirectory(projectRoot, segments) {
  const projectRealPath = await realpath(projectRoot);
  let current = projectRealPath;

  for (const segment of segments) {
    if (typeof segment !== "string" || segment === "." || segment === ".." || segment.includes(path.sep)) {
      throw new Error("directory segment contains unsupported characters");
    }

    const target = path.join(current, segment);
    let stat = await lstatIfPresent(target);
    if (stat?.isSymbolicLink()) throw new Error(`${path.relative(projectRealPath, target)} must not be a symbolic link`);
    if (stat && !stat.isDirectory()) throw new Error(`${path.relative(projectRealPath, target)} must be a directory`);
    if (!stat) {
      await mkdir(target);
      stat = await lstat(target);
    }
    if (!stat.isDirectory()) throw new Error(`${path.relative(projectRealPath, target)} must be a directory`);

    const targetRealPath = await realpath(target);
    if (!isPathInside(projectRealPath, targetRealPath)) throw new Error("directory escapes the project");
    current = targetRealPath;
  }

  return current;
}
