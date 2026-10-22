const DEFAULT_TIMEOUT_MS = 240_000;

export function parseErrorDetail(payload, fallback = 'The detector could not complete this request.') {
  if (!payload || typeof payload !== 'object') return fallback;

  const {detail} = payload;
  if (typeof detail === 'string' && detail.trim()) return detail;

  if (Array.isArray(detail)) {
    const messages = detail.map(item => {
      if (!item || typeof item !== 'object') return '';
      const location = Array.isArray(item.loc)
        ? item.loc.filter(part => part !== 'body').join('.')
        : '';
      const message = typeof item.msg === 'string' ? item.msg : '';
      return location && message ? `${location}: ${message}` : message;
    }).filter(Boolean);
    if (messages.length) return messages.join('; ');
  }

  if (typeof payload.message === 'string' && payload.message.trim()) return payload.message;
  return fallback;
}

export function normalizeResult(payload) {
  if (!payload || typeof payload !== 'object' || !Array.isArray(payload.claims)) {
    throw new Error('The detector returned an unexpected response. Check the API logs and try again.');
  }
  return {
    ...payload,
    warnings: Array.isArray(payload.warnings) ? payload.warnings : [],
    claims: payload.claims.map(claim => ({
      ...(claim && typeof claim === 'object' ? claim : {}),
      evidence: Array.isArray(claim?.evidence) ? claim.evidence : [],
    })),
  };
}

async function readPayload(response) {
  const contentType = response.headers?.get?.('content-type') || '';
  if (contentType.includes('application/json')) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  const text = await response.text().catch(() => '');
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return {detail: text};
  }
}

export async function detect(payload, {signal, timeoutMs = DEFAULT_TIMEOUT_MS} = {}) {
  const controller = new AbortController();
  let timedOut = false;

  const abortFromCaller = () => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener('abort', abortFromCaller, {once: true});
  }
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const response = await fetch('/detect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const body = await readPayload(response);
    const requestId = response.headers?.get?.('x-request-id');
    if (!response.ok) {
      const fallback = response.status >= 500
        ? 'The detector is unavailable. Check the API logs and model dependencies.'
        : `The detector rejected this request (${response.status}).`;
      const message = parseErrorDetail(body, fallback);
      throw new Error(requestId ? `${message} Request ID: ${requestId}` : message);
    }
    return normalizeResult(body);
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw new Error(timedOut
        ? 'The request timed out. The first run may need about 60 seconds to load models.'
        : 'The request was cancelled.');
    }
    if (error instanceof TypeError) {
      throw new Error('Could not reach the detector API. Is the backend running on 127.0.0.1:8000?');
    }
    throw error;
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', abortFromCaller);
  }
}
