import { existsSync } from 'node:fs';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const local = fileURLToPath(new URL('../.venv/bin/python', import.meta.url));
const python = process.env.VOICE_AGENT_PYTHON || (existsSync(local) ? local : 'python3');
const env = { ...process.env, PYTHONDONTWRITEBYTECODE: '1' };
// Some macOS Python installs have no default CA file. Use a trusted bundle,
// preserving TLS verification and any certificate configuration set by the user.
if (!env.SSL_CERT_FILE && !env.SSL_CERT_DIR) {
  const probe = spawnSync(python, ['-c', 'import ssl,certifi; print("" if ssl.get_default_verify_paths().cafile else certifi.where())'], { encoding: 'utf8' });
  const ca = probe.status === 0 ? probe.stdout.trim() : '';
  if (ca && existsSync(ca)) env.SSL_CERT_FILE = ca;
}
const child = spawn(python, process.argv.slice(2), {
  cwd: root, stdio: 'inherit', env,
});
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
child.on('error', () => { process.stderr.write('Python unavailable; create .venv and install backend/requirements.txt.\n'); process.exitCode = 1; });
child.on('exit', (code) => { process.exitCode = code ?? 1; });
