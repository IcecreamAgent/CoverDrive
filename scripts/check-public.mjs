import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFile, lstat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const allowedRoots = new Set(['backend', 'frontend', 'scripts', '.agents', '.codex', 'third_party']);
const allowedFiles = new Set(['.gitignore', '.env.example', 'AGENTS.md', 'PROJECT_PROFILE.md', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'package.json', 'package-lock.json']);

export function sensitiveFindings(name, text) {
  const findings = [];
  const rules = [
    ['private-key', /-----BEGIN (?:OPENSSH|RSA|EC|DSA|ENCRYPTED)? ?PRIVATE KEY-----/],
    ['github-token', /\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b/],
    ['service-api-key', /\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b/],
    ['cloud-access-key', /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/],
    ['configured-model-id', /\b(?:doubao|deepseek|qwen|gpt)[-_][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*/i],
    ['fixed-native-model', /["']\d+\.\d+\.\d+\.\d+["']/],
    ['personal-path', /\/(?:Users|home)\/[A-Za-z0-9_.-]+\//],
    ['private-deployment', /\b[a-z0-9.-]+\.icecream\.xin\b/i],
  ];
  for (const [index, line] of text.split('\n').entries()) {
    for (const [category, pattern] of rules) {
      if (pattern.test(line)) findings.push(`${name}:${index + 1}: ${category}`);
    }
    // Detect credential-shaped constant assignments without reporting their values.
    if (/\b(?:api_?key|access_?key|app_?key|secret|token)\s*["']?\s*[:=]/i.test(line)) {
      for (const match of line.matchAll(/["']([A-Za-z0-9_+\/=.-]{16,})["']/g)) {
        const value = match[1];
        if (/[A-Z]/.test(value) && /[a-z]/.test(value) && /\d/.test(value)
            && !/^(?:fake|test|isolated|example|private|configured|sentinel|own)[-_]/i.test(value)
            && !/^[A-Z0-9_]+$/.test(value)) {
          findings.push(`${name}:${index + 1}: credential-constant`);
        }
      }
    }
  }
  return findings;
}

export async function checkPublic() {
  const result = spawnSync('git', ['ls-files', '--cached', '--others', '--exclude-standard', '-z'], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 0, 'Run the public check from a Git checkout.');
  const files = [...new Set(result.stdout.split('\0').filter(Boolean))];
  const failures = [];
  for (const name of files) {
    const parts = name.split('/');
    if (!(parts.length === 1 ? allowedFiles.has(name) : allowedRoots.has(parts[0]))) failures.push(`${name}: outside public allowlist`);
    if (parts.some(part => ['.env', 'node_modules', 'dist', '__pycache__', '.venv', 'runs', 'tasks', '.claude', '.playwright-mcp'].includes(part))
        || /\.(?:pyc|pyo|log|sqlite3?|db|xlsx|csv|pem|key|p12|pfx|har|wav|pcm|mp3|zip)$/i.test(name)
        || (path.basename(name).startsWith('.env.') && name !== '.env.example')) failures.push(`${name}: private or generated file`);
    const stat = await lstat(path.join(root, name));
    if (!stat.isFile()) { failures.push(`${name}: expected a regular file`); continue; }
    if (name === 'backend/models/silero_vad.onnx') continue;
    const buffer = await readFile(path.join(root, name));
    if (buffer.includes(0)) { failures.push(`${name}: unexpected binary file`); continue; }
    failures.push(...sensitiveFindings(name, buffer.toString('utf8')));
  }
  for (const name of allowedFiles) assert(files.includes(name), `Missing public file: ${name}`);
  const example = await readFile(path.join(root, '.env.example'), 'utf8');
  for (const key of ['LLM_BASE_URL', 'LLM_MODEL_ID', 'LLM_API_KEY', 'SEEDUPLEX_URL', 'SEEDUPLEX_MODEL', 'SEEDUPLEX_SPEAKER', 'SEEDUPLEX_API_KEY', 'SEEDUPLEX_APP_ID', 'SEEDUPLEX_ACCESS_KEY', 'SEEDUPLEX_APP_KEY']) {
    assert(example.split('\n').includes(`${key}=`), `Example must require user-supplied ${key}`);
  }
  assert(example.includes('INSURANCE_WRITES_ENABLED=false'));
  const readme = await readFile(path.join(root, 'README.md'), 'utf8');
  for (const key of ['LLM_MODEL_ID', 'LLM_API_KEY', 'SEEDUPLEX_MODEL', 'SEEDUPLEX_API_KEY']) assert(readme.includes(key), `README must explain ${key}`);
  assert((await readFile(path.join(root, 'LICENSE'), 'utf8')).startsWith('MIT License'));
  assert.equal(failures.length, 0, 'Public source checks failed (values redacted):\n' + failures.join('\n'));
  console.log(`Public source check passed: ${files.length} allowlisted files; blank model/key example; MIT license; no detected sensitive markers.`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  checkPublic().catch(error => { console.error(error.message); process.exitCode = 1; });
}
