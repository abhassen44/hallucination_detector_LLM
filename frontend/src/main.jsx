import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const labels = {SUPPORTED: 'Supported', CONTRADICTED: 'Contradicted', INSUFFICIENT: 'Insufficient evidence'};

function Badge({label}) {
  return <span className={`badge ${label}`}>{labels[label] || label}</span>;
}

function NLIBar({nli}) {
  if (!nli) return null;
  const total = nli.entailment + nli.neutral + nli.contradiction;
  if (total <= 0) return null;
  const w = 80;
  return (
    <span className="nli-bar" title={`E: ${(nli.entailment*100).toFixed(1)}%  N: ${(nli.neutral*100).toFixed(1)}%  C: ${(nli.contradiction*100).toFixed(1)}%`}>
      <span className="bar-segment" style={{width: `${(nli.entailment/total)*w}px`, background: 'var(--supported)'}}/>
      <span className="bar-segment" style={{width: `${(nli.neutral/total)*w}px`, background: 'var(--insufficient)'}}/>
      <span className="bar-segment" style={{width: `${(nli.contradiction/total)*w}px`, background: 'var(--contradicted)'}}/>
    </span>
  );
}

function ClaimCard({claim, index}) {
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
        <summary>Inspect {claim.evidence.length} evidence passage{claim.evidence.length === 1 ? '' : 's'}</summary>
        {claim.evidence.map((ev, j) => (
          <blockquote key={j}>
            <p>{ev.text}</p>
            <footer>
              {ev.source === 'provided'
                ? 'Your source text'
                : `${ev.page_id?.replaceAll('_', ' ')} · sentence ${ev.sent_id}`}
              {ev.nli && (
                <span>
                  E {(ev.nli.entailment * 100).toFixed(1)}%
                  · N {(ev.nli.neutral * 100).toFixed(1)}%
                  · C {(ev.nli.contradiction * 100).toFixed(1)}%
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
        {result.extraction_model}
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

  async function verify(e) {
    e.preventDefault();
    setBusy(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch('/detect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          question: question.trim() || null,
          answer,
          method,
          top_k: 5,
          evidence: source.trim() ? [source.trim()] : null,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(typeof data.detail === 'string'
          ? data.detail
          : 'Check the answer length and selected retrieval method.');
      }
      setResult(data);
    } catch (err) {
      setError(err.message || 'Could not connect to the detector.');
    } finally {
      setBusy(false);
    }
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

            <label htmlFor="answer">Answer to verify</label>
            <textarea
              id="answer" rows={7} required maxLength={12000} value={answer}
              onChange={e => setAnswer(e.target.value)} disabled={busy}
              placeholder="Paste the LLM answer you want to fact-check…"
            />

            <label htmlFor="source">Source text <small>optional</small></label>
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

            <button disabled={busy || !answer.trim()}>
              {busy ? 'Inspecting claims…' : 'Verify answer →'}
            </button>

            <p className="note">
              {source.trim()
                ? 'First use may take ~30s while DeBERTa loads.'
                : 'First use may take ~60s while models and indexes load.'}
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
