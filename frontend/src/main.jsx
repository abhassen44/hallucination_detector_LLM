import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const names = {SUPPORTED: 'Supported', CONTRADICTED: 'Contradicted', INSUFFICIENT: 'Insufficient evidence'};
function Badge({label}) {return <span className={`badge ${label}`}>{names[label] || label}</span>}
function App() {
  const [question, setQuestion] = useState('How many times has Chris Paul led the NBA in assists and steals?');
  const [answer, setAnswer] = useState('Chris Paul is a four-time NBA assist leader and six-time steal leader.');
  const [source, setSource] = useState('');
  const [method, setMethod] = useState('hybrid');
  const [busy, setBusy] = useState(false);
  const [rerank, setRerank] = useState(false);
  const [groundedReview, setGroundedReview] = useState(false);
  const [rerankReady, setRerankReady] = useState(false);
  useEffect(() => {
    fetch('/health').then(r => r.json()).then(data => setRerankReady(Boolean(data.reranker_available))).catch(() => {});
  }, []);
  const [evidencePolicy, setEvidencePolicy] = useState('all');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  async function verify(e) {
    e.preventDefault(); setBusy(true); setError(''); setResult(null);
    try {
      const response = await fetch('/detect', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: question.trim() || null, answer, method, top_k: 5, rerank: rerank && rerankReady && !source.trim(),
          evidence_policy: source.trim() ? 'all' : evidencePolicy, grounded_review: groundedReview,
          evidence: source.trim() ? [source.trim()] : null})});
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the answer length and selected retrieval method.');
      setResult(data);
    } catch (err) {setError(err.message || 'Could not connect to the detector.');}
    finally {setBusy(false);}
  }
  return <main>
    <header><div className="eyebrow">EVIDENCE / CLAIMS / VERIFICATION</div><h1>Claim Inspector<span>.</span></h1>
      <p>Break an answer into factual claims and inspect the evidence behind each one.</p></header>
    <div className="layout"><section className="panel"><h2>Check an answer</h2>
      <form onSubmit={verify}>
        <label htmlFor="question">Question <small>optional</small></label>
        <input id="question" maxLength={12000} value={question} onChange={e => setQuestion(e.target.value)} disabled={busy}/>
        <label htmlFor="answer">Answer to verify</label>
        <textarea id="answer" rows={8} required maxLength={12000} value={answer} onChange={e => setAnswer(e.target.value)} disabled={busy}/>
        <label htmlFor="source">Source text <small>optional</small></label>
        <textarea id="source" rows={4} maxLength={12000} value={source} onChange={e => setSource(e.target.value)} disabled={busy} placeholder="Paste a passage to check the answer against it."/>
        <p className="note">With source text, claims are checked against your passage. Otherwise, evidence is retrieved from a limited Wikipedia collection. Use a concise passage containing the relevant facts.</p>
        <label htmlFor="method">Evidence search</label>
        <select id="method" value={method} onChange={e => setMethod(e.target.value)} disabled={busy || Boolean(source.trim())}>
          <option value="hybrid">Hybrid · keyword + semantic</option><option value="dense">Semantic search</option><option value="bm25">Keyword search</option>
        </select>
        <label htmlFor="evidence-policy">Evidence to use</label>
        <select id="evidence-policy" value={source.trim() ? 'all' : evidencePolicy} onChange={e => setEvidencePolicy(e.target.value)} disabled={busy || Boolean(source.trim())}>
          <option value="all">All retrieved passages · baseline</option>
          <option value="top1">First retrieved passage · experimental</option>
        </select>
        {evidencePolicy !== 'all' && !source.trim() && <p className="note">Promising on validation data; accuracy on new examples has not been established.</p>}
        <label><input type="checkbox" checked={rerank} onChange={e => setRerank(e.target.checked)} disabled={busy || Boolean(source.trim()) || !rerankReady}/> Rerank search results <small>experimental · slower</small></label>
        {!rerankReady && <p className="note">Optional reranking is unavailable until its model is installed. Evidence search and supplied-source checks are ready.</p>}
        <label><input type="checkbox" checked={groundedReview} onChange={e => setGroundedReview(e.target.checked)} disabled={busy}/> Double-check contradictions <small>experimental · slower</small></label>
        <p className="note">An optional second model checks the passages before confirming a contradiction or resolving conflicting signals.</p>
        <button disabled={busy || !answer.trim()}>{busy ? 'Inspecting claims…' : 'Verify answer →'}</button>
        <p className="note">{source.trim() ? 'First use may take longer while the verification model loads.' : 'First use may take longer while models and the evidence index load.'}</p>
      </form></section>
      <section aria-live="polite" aria-busy={busy}>
        {error && <div role="alert" className="error">{error}</div>}
        {busy && <div className="empty">{source.trim() ? 'Extracting claims and checking your source text…' : 'Extracting claims and checking Wikipedia evidence…'}</div>}
        {!busy && !result && !error && <div className="empty"><span className="symbol">⌕</span><h2>Follow the evidence</h2><p>Your claim breakdown will appear here.</p></div>}
        {result && <><div className="summary"><h2>Answer assessment</h2><Badge label={result.label}/>{result.decision?.status === 'abstain' && <p className="warning">Review required: {result.decision.reasons.map(r => r.replaceAll('_', ' ')).join(', ')}. The displayed verdict is provisional.</p>}<p>{result.claims.length} claims inspected · {result.evidence_source === 'provided' ? 'Your source text' : 'Wikipedia evidence'} · {result.extraction_model}{result.reranked ? ' · Reranked' : ''}</p></div>
          {result.warnings.map((w,i) => <p className="warning" key={i}>{w}</p>)}
          {result.claims.map((claim,i) => <article className="panel claim" key={i}>
            <div className="claim-heading"><small>CLAIM {String(i+1).padStart(2, '0')}</small><Badge label={claim.label}/></div>
            <h3>{claim.claim}</h3>
            {claim.label === 'INSUFFICIENT' && <p className="note">These passages do not establish this claim. This alone does not prove the claim is false.</p>}
            {claim.evidence_conflict && <p className="warning">Strong support and contradiction signals appear in these passages. Review their relevance before relying on this label.</p>}
            {claim.grounded_review && claim.grounded_review.status !== 'not_required' && <div className="note"><strong>Second evidence check</strong><p>{claim.grounded_review.reason}</p>{claim.baseline_label !== claim.label && <p>Initial verdict: {names[claim.baseline_label]}. The final verdict remains unconfirmed.</p>}{claim.grounded_review.citations?.map((c,k) => <blockquote key={k}>Passage {c.evidence_id}: “{c.quote}”</blockquote>)}</div>}
            <details><summary>Inspect {claim.evidence.length} evidence passage{claim.evidence.length === 1 ? '' : 's'}</summary>
              {claim.evidence.map((ev,j) => <blockquote key={j}><p>{ev.text}</p><footer>{ev.used_for_verdict === false ? 'Not used for verdict · ' : ''}{ev.source === 'provided' ? 'Your source text' : `${ev.page_id?.replaceAll('_', ' ')} · sentence ${ev.sent_id}`}
                {ev.nli && <span>Entailment {(ev.nli.entailment*100).toFixed(1)}% · Contradiction {(ev.nli.contradiction*100).toFixed(1)}%</span>}</footer></blockquote>)}
            </details>
          </article>)}</>}
      </section></div>
    <footer className="footnote">Results use a fixed-threshold NLI baseline with optional second-model evidence review, and supplied text or a limited Wikipedia corpus. Evidence scores are model signals, not calibrated probabilities of truth. Human review remains useful.</footer>
  </main>
}
createRoot(document.getElementById('root')).render(<App/>);
