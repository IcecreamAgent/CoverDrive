import test from 'node:test';
import assert from 'node:assert/strict';
import { sensitiveFindings } from './check-public.mjs';

test('scanner reports location/category while withholding private material', () => {
  const fakeMaterial = ['ghp_', 'A'.repeat(36)].join('');
  const result = sensitiveFindings('fixture.txt', fakeMaterial);
  assert.deepEqual(result, ['fixture.txt:1: github-token']);
  assert(!JSON.stringify(result).includes(fakeMaterial));
});

test('scanner rejects private key headers and deployed model identifiers', () => {
  assert.equal(sensitiveFindings('fixture.txt', ['-----BEGIN ', 'OPENSSH PRIVATE KEY-----'].join('')).length, 1);
  assert.equal(sensitiveFindings('fixture.txt', ['deep', 'seek-v999-test'].join('')).length, 1);
  assert.equal(sensitiveFindings('fixture.txt', ['api_', 'key = ', '"A9bcDef01234567890123456789"'].join('')).length, 1);
});

test('blank variables, explicit fake credentials and role labels are allowed', () => {
  assert.deepEqual(sensitiveFindings('.env.example', 'LLM_API_KEY=\nSEEDUPLEX_MODEL='), []);
  assert.deepEqual(sensitiveFindings('fixture.py', 'api_key = "fake-PrivateKey123456"'), []);
  assert.deepEqual(sensitiveFindings('fixture.py', 'model = "任务 LLM"'), []);
});


test('only authorized demo links in README are public; other deployment material remains rejected', () => {
  const demo = 'https://' + 'coverdrive' + '.icecream.xin';
  const backend = 'https://' + 'cdbackend' + '.icecream.xin';
  assert.deepEqual(sensitiveFindings('README.md', `[voice](${demo}) [backend](${backend})`), []);
  assert.equal(sensitiveFindings('fixture.py', `[voice](${demo})`).length, 1);
  assert.equal(sensitiveFindings('README.md', demo + '/private').length, 1);
  assert.equal(sensitiveFindings('README.md', 'https://' + 'internal' + '.icecream.xin').length, 1);
  const credential = ['ghp_', 'A'.repeat(36)].join('');
  assert.deepEqual(sensitiveFindings('README.md', `[voice](${demo}) ${credential}`), ['README.md:1: github-token']);
});
