import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { detect } from './api.js';
import './style.css';

const labels = {SUPPORTED: 'Supported', CONTRADICTED: 'Contradicted', INSUFFICIENT: 'Insufficient evidence'};

function Badge({label}) {
  return <span className={`badge ${label}`}>{labels[label] || label}</span>;
}

function NLIBar({nli}) {
  if (!nli) return null;
  const e = Number(nli.entailment) || 0;
  const n = Number(nli.neutral) || 0;
  const c = Number(nli.contradiction) || 0;
  const total = e + n + c;
  if (!Number.isFinite(total) || total <= 0) return null;
  const w = 80;
  return (
    <span className="nli-bar" title={`E: ${(e*100).toFixed(1)}%  N: ${(n*100).toFixed(1)}%  C: ${(c*100).toFixed(1)}%`}>
      <span className="bar-segment" style={{width: `${(e/total)*w}px`, background: 'var(--supported)'}}/>
      <span className="bar-segment" style={{width: `${(n/total)*w}px`, background: 'var(--insufficient)'}}/>
      <span className="bar-segment" style={{width: `${(c/total)*w}px`, background: 'var(--contradicted)'}}/>
    </span>
  );
}

function ClaimCard({claim, index}) {
  const evidence = Array.isArray(claim.evidence) ? claim.evidence : [];
  return (
    <article className="panel claim" style={{animationDelay: `${0.05 * index}s`}}>
      <div className="claim-heading">
        <small>CLAIM {String(index + 1).padStart(2, '0')}</small>
        <Badge label={claim.label}/>
      </div>
      <h3>{claim.claim}</h3>
      {claim.evidence_conflict && (
        <div className="warning">
          Strong support and contradiction signals appear in these passages. Review their relevance before relying on this label.
        </div>
      )}
      <details>
        <summary>Inspect {evidence.length} evidence passage{evidence.length === 1 ? '' : 's'}</summary>
        {evidence.length === 0 && <p className="note">No evidence passages were returned for this claim.</p>}
        {evidence.map((ev, j) => (
          <blockquote key={j}>
            <p>{ev.text}</p>
            <footer>
              {ev.source === 'provided'
                ? 'Your source text'
                : `${ev.page_id?.replaceAll('_', ' ') ?? 'Unknown source'} · sentence ${ev.sent_id ?? '?'}`}
              {ev.nli && (
                <span>
                  E {((Number(ev.nli.entailment) || 0) * 100).toFixed(1)}%
                  · N {((Number(ev.nli.neutral) || 0) * 100).toFixed(1)}%
                  · C {((Number(ev.nli.contradiction) || 0) * 100).toFixed(1)}%
                  <NLIBar nli={ev.nli}/>
                </span>
              )}
            </footer>
          </blockquote>
        ))}
      </details>
    </article>
  );
}

function ResultsSummary({result}) {
  const supported = result.claims.filter(c => c.label === 'SUPPORTED').length;
  const contradicted = result.claims.filter(c => c.label === 'CONTRADICTED').length;
  const insufficient = result.claims.filter(c => c.label === 'INSUFFICIENT').length;
  return (
    <div className="summary">
      <h2>Assessment</h2>
      <Badge label={result.label}/>
      <p>
        {result.claims.length} claims inspected ·{' '}
        {result.evidence_source === 'provided' ? 'Your source text' : 'Wikipedia evidence'} ·{' '}
        {result.extraction_model || 'claim extractor'}
      </p>
      <div className="summary-stats">
        <div className="stat">
          <strong style={{color: 'var(--supported)'}}>{supported}</strong>
          Supported
        </div>
        <div className="stat">
          <strong style={{color: 'var(--contradicted)'}}>{contradicted}</strong>
          Contradicted
        </div>
        <div className="stat">
          <strong style={{color: 'var(--insufficient)'}}>{insufficient}</strong>
          Insufficient
        </div>
      </div>
    </div>
  );
}

function LoadingState({hasSource}) {
  return (
    <div className="loading-container">
      <div className="spinner"/>
      <div className="loading-text">
        {hasSource ? 'Verifying claims against your source…' : 'Searching Wikipedia evidence…'}
      </div>
      <div className="loading-step">
        Extracting claims → Retrieving evidence → Running NLI verification
      </div>
    </div>
  );
}

function App() {
  const [question, setQuestion] = useState('How many times has Chris Paul led the NBA in assists and steals?');
  const [answer, setAnswer] = useState('Chris Paul is a four-time NBA assist leader and six-time steal leader.');
  const [source, setSource] = useState('');
  const [method, setMethod] = useState('hybrid');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const abortRef = useRef(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  async function verify(e) {
    e.preventDefault();
    if (!answer.trim()) {
      setError('Enter an answer to verify.');
      return;
    }
    setBusy(true);
    setError('');
    setResult(null);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const data = await detect({
        question: question.trim() || null,
        answer: answer.trim(),
        method,
        top_k: 5,
        evidence: source.trim() ? [source.trim()] : null,
      }, {signal: controller.signal});
      setResult(data);
    } catch (err) {
      setError(err.message || 'Could not connect to the detector.');
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setBusy(false);
    }
  }

  function cancel() {
    abortRef.current?.abort();
  }

  return (
    <main>
      <header>
        <div className="eyebrow">Claim-Level Verification</div>
        <h1>Claim Inspector</h1>
        <p>
          Break an LLM answer into atomic factual claims. Retrieve evidence from Wikipedia
          or your own source text. Verify each claim with DeBERTa NLI.
        </p>
      </header>

      <div className="layout">
        <section className="panel">
          <h2>Check an answer</h2>
          <form onSubmit={verify}>
            <label htmlFor="question">Question <small>optional</small></label>
            <input
              id="question" maxLength={12000} value={question}
              onChange={e => setQuestion(e.target.value)} disabled={busy}
              placeholder="What question was the answer responding to?"
            />

            <label htmlFor="answer">Answer to verify <small>{answer.length.toLocaleString()} / 12,000</small></label>
            <textarea
              id="answer" rows={7} required maxLength={12000} value={answer}
              onChange={e => setAnswer(e.target.value)} disabled={busy}
              placeholder="Paste the LLM answer you want to fact-check…"
            />

            <label htmlFor="source">Source text <small>optional · {source.length.toLocaleString()} / 12,000</small></label>
            <textarea
              id="source" rows={4} maxLength={12000} value={source}
              onChange={e => setSource(e.target.value)} disabled={busy}
              placeholder="Paste a passage to verify claims against. Otherwise Wikipedia is used."
            />

            <p className="note">
              {source.trim()
                ? 'Claims will be checked against your passage using NLI.'
                : 'Without source text, evidence is retrieved from a 248k-sentence Wikipedia corpus using BM25 + FAISS hybrid search.'}
            </p>

            <label htmlFor="method">Evidence search</label>
            <select
              id="method" value={method}
              onChange={e => setMethod(e.target.value)}
              disabled={busy || Boolean(source.trim())}
            >
              <option value="hybrid">Hybrid · keyword + semantic (RRF)</option>
              <option value="dense">Dense · semantic search (FAISS)</option>
              <option value="bm25">Sparse · keyword search (BM25)</option>
            </select>

            <div className="button-row">
              <button disabled={busy || !answer.trim()}>
                {busy ? 'Inspecting claims…' : 'Verify answer →'}
              </button>
              {busy && (
                <button type="button" className="secondary" onClick={cancel}>
                  Cancel
                </button>
              )}
            </div>

            <p className="note">
              {source.trim()
                ? 'First use may take ~30s while DeBERTa loads. Cancel the request if it stalls.'
                : 'First use may take ~60s while models and indexes load. Cancel the request if it stalls.'}
            </p>
          </form>
        </section>

        <section aria-live="polite" aria-busy={busy}>
          {error && <div role="alert" className="error">{error}</div>}

          {busy && <LoadingState hasSource={Boolean(source.trim())}/>}

          {!busy && !result && !error && (
            <div className="empty">
              <span className="symbol">🔍</span>
              <h2>Follow the evidence</h2>
              <p>Your claim breakdown and verification results will appear here.</p>
            </div>
          )}

          {result && (
            <>
              <ResultsSummary result={result}/>
              {result.claims.length === 0 && (
                <div className="warning">
                  No atomic claims were extracted. Try a shorter, more factual answer or inspect the extraction configuration.
                </div>
              )}
              {result.warnings.map((w, i) => (
                <div className="warning" key={i}>{w}</div>
              ))}
              {result.claims.map((claim, i) => (
                <ClaimCard key={i} claim={claim} index={i}/>
              ))}
            </>
          )}
        </section>
      </div>

      <footer className="footnote">
        Results use a fixed-threshold DeBERTa-v3-base-mnli baseline. Evidence scores are NLI model signals,
        not calibrated probabilities. The Wikipedia corpus covers ~248k sentences from FEVER pages; coverage
        is limited for topics outside that set. Learned v5 research candidate (final-policy macro-F1 0.737;
        NEI answer precision remains poor) exists but is not served in the demo. Human review remains useful. Read the{' '}
        <a href="https://github.com/abhassen44/hallucination_detector_LLM" style={{color: 'var(--accent)'}}>
          documentation
        </a>{' '}
        for methodology and limitations.
      </footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App/>);
