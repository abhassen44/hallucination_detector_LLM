import assert from 'node:assert/strict';
import test from 'node:test';

import {detect, normalizeResult, parseErrorDetail} from './api.js';


function response({status = 200, body = {}, contentType = 'application/json', requestId = 'req-1'} = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get(name) {
        if (name.toLowerCase() === 'content-type') return contentType;
        if (name.toLowerCase() === 'x-request-id') return requestId;
        return null;
      },
    },
    async json() {
      return body;
    },
    async text() {
      return typeof body === 'string' ? body : JSON.stringify(body);
    },
  };
}


test('formats string and validation-list error details', () => {
  assert.equal(parseErrorDetail({detail: 'No evidence found.'}), 'No evidence found.');
  assert.equal(
    parseErrorDetail({detail: [
      {loc: ['body', 'answer'], msg: 'Field required'},
      {loc: ['body', 'top_k'], msg: 'Input should be less than or equal to 20'},
    ]}),
    'answer: Field required; top_k: Input should be less than or equal to 20',
  );
  assert.equal(parseErrorDetail(null, 'fallback'), 'fallback');
});


test('normalizes missing warnings and evidence', () => {
  assert.deepEqual(
    normalizeResult({label: 'INSUFFICIENT', claims: [{claim: 'a', label: 'INSUFFICIENT'}]}),
    {
      label: 'INSUFFICIENT',
      claims: [{claim: 'a', label: 'INSUFFICIENT', evidence: []}],
      warnings: [],
    },
  );
  assert.throws(() => normalizeResult({label: 'SUPPORTED'}), /unexpected response/);
});


test('detect returns a normalized successful response', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => response({
    body: {label: 'SUPPORTED', claims: [{claim: 'Paris is in France.', label: 'SUPPORTED'}]},
  });
  try {
    const result = await detect({answer: 'Paris is in France.'});
    assert.equal(result.label, 'SUPPORTED');
    assert.deepEqual(result.warnings, []);
    assert.deepEqual(result.claims[0].evidence, []);
  } finally {
    globalThis.fetch = originalFetch;
  }
});


test('detect surfaces API detail and request id', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => response({
    status: 503,
    body: {detail: 'Models are unavailable.'},
    requestId: 'support-123',
  });
  try {
    await assert.rejects(
      detect({answer: 'question'}),
      /Models are unavailable\. Request ID: support-123/,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});


test('detect reports a friendly network error', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError('fetch failed');
  };
  try {
    await assert.rejects(detect({answer: 'question'}), /backend running/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});


test('detect handles a non-JSON error body', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => response({
    status: 502,
    body: 'Bad gateway',
    contentType: 'text/html',
  });
  try {
    await assert.rejects(detect({answer: 'question'}), /Bad gateway/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});


test('detect times out and can be cancelled', async () => {
  const originalFetch = globalThis.fetch;
  const hangingFetch = (_url, options) => new Promise((_, reject) => {
    options.signal.addEventListener('abort', () => {
      const error = new Error('aborted');
      error.name = 'AbortError';
      reject(error);
    }, {once: true});
  });
  globalThis.fetch = hangingFetch;
  try {
    await assert.rejects(detect({answer: 'slow'}, {timeoutMs: 10}), /timed out/);

    const controller = new AbortController();
    const pending = detect({answer: 'cancel'}, {signal: controller.signal});
    controller.abort();
    await assert.rejects(pending, /cancelled/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
