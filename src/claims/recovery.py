"""Conservative syntax-recovery proposals for review, never automatic dataset edits."""
import json
import re


def propose_terminal_quote_recovery(raw):
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.I).strip()
    try:
        json.loads(cleaned)
        return None  # Already valid: do not rewrite it.
    except ValueError:
        pass
    if not cleaned.startswith('['):
        return None
    match = re.search(r'\\"(\]+)$', cleaned)
    if not match:
        return None
    # Only add a JSON string terminator at the final array boundary. This keeps
    # every model-produced word, and does not invent or correct factual content.
    pos = match.start(1)
    candidate = cleaned[:pos]+'"'+cleaned[pos:]
    try:
        value, end = json.JSONDecoder().raw_decode(candidate)
    except ValueError:
        return None
    tail = candidate[end:].strip()
    if tail not in ('', ']') or not isinstance(value, list) or not value:
        return None
    if not all(isinstance(v, str) and v.strip() for v in value):
        return None
    return {'proposed_claims': value, 'added_string_delimiters': 1,
            'removed_extra_array_closers': len(tail), 'applied_to_dataset': False}
