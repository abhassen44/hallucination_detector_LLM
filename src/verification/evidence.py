"""Deterministic evidence policies shared by serving and development studies."""
from __future__ import annotations

MODES = ('all', 'top1', 'top3', 'same_page')


def select_evidence(evidence: list[dict], mode: str = 'all') -> list[dict]:
    if mode not in MODES:
        raise ValueError(f'Unknown evidence policy: {mode}')
    if mode == 'top1':
        return evidence[:1]
    if mode == 'top3':
        return evidence[:3]
    if mode == 'same_page' and evidence:
        page = evidence[0].get('page_id')
        if page is None:
            raise ValueError('same_page requires page provenance')
        return [hit for hit in evidence if hit.get('page_id') == page]
    return list(evidence)


def acceptance(label: str, *, has_evidence: bool, conflict: bool) -> dict:
    """Operational review decision, not a calibrated confidence threshold."""
    reasons = []
    if not has_evidence:
        reasons.append('no_evidence')
    elif label == 'INSUFFICIENT':
        reasons.append('insufficient_evidence')
    if conflict:
        reasons.append('conflicting_evidence')
    return {'status': 'abstain' if reasons else 'answer', 'reasons': reasons,
            'policy': 'evidence_review_v1'}
