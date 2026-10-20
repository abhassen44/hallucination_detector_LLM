import pytest

from src.claims.recovery import propose_terminal_quote_recovery


@pytest.mark.parametrize('raw,expected,extra', [
    (r'["Derek Sherinian is an American journalist.", "Derek Sherinian has been the Literary Editor of \"The Weekly Standard.\"]',
     ['Derek Sherinian is an American journalist.', 'Derek Sherinian has been the Literary Editor of "The Weekly Standard."'], 0),
    (r'["Kato Khandwala helped produce the album \"I Brought You My Bullets, You Brought Me Your Love.\"]]',
     ['Kato Khandwala helped produce the album "I Brought You My Bullets, You Brought Me Your Love."'], 1),
])
def test_terminal_repair_preserves_model_words_and_is_proposal_only(raw,expected,extra):
    result = propose_terminal_quote_recovery(raw)
    assert result['proposed_claims'] == expected
    assert result['removed_extra_array_closers'] == extra
    assert result['applied_to_dataset'] is False


def test_other_output_is_not_rewritten():
    assert propose_terminal_quote_recovery('["already valid"]') is None
    assert propose_terminal_quote_recovery('["broken') is None
    assert propose_terminal_quote_recovery('not json') is None
