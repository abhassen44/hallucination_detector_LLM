import pytest

from src.datasets.schema import CONTRADICTED, INSUFFICIENT, SUPPORTED
from src.evaluation.decomposition import decompose


def test_buckets_partition_pipeline_errors():
    gold = [SUPPORTED, SUPPORTED, CONTRADICTED, CONTRADICTED, SUPPORTED, CONTRADICTED]
    pred = [SUPPORTED, INSUFFICIENT, CONTRADICTED, SUPPORTED, CONTRADICTED, INSUFFICIENT]
    oracle = [SUPPORTED, SUPPORTED, CONTRADICTED, CONTRADICTED, SUPPORTED, SUPPORTED]
    stats = decompose(gold, pred, oracle)

    buckets = stats['error_buckets']
    assert stats['n'] == 6
    assert stats['answered'] == 4 and stats['abstained'] == 2
    assert stats['pipeline_error'] == 4
    assert buckets['abstained_oracle_correct'] == 1   # claim 1: answer was recoverable
    assert buckets['abstained_oracle_wrong'] == 1     # claim 5: oracle also wrong
    assert buckets['confident_oracle_correct'] == 2   # claims 3 and 4: pipeline introduced them
    assert buckets['confident_oracle_wrong'] == 0
    assert buckets['pipeline_wins_over_oracle'] == 0
    assert stats['oracle_error'] == 1                 # claim 5
    assert stats['error_buckets_sum'] == stats['pipeline_error']
    assert stats['net_incremental_error_vs_oracle'] == 3


def test_pipeline_win_is_counted_separately():
    gold = [SUPPORTED]
    pred = [SUPPORTED]
    oracle = [CONTRADICTED]
    stats = decompose(gold, pred, oracle)
    assert stats['pipeline_error'] == 0
    assert stats['error_buckets']['pipeline_wins_over_oracle'] == 1
    assert stats['net_incremental_error_vs_oracle'] == -1


def test_selective_accuracy_only_uses_answered_claims():
    gold = [SUPPORTED, CONTRADICTED, SUPPORTED]
    pred = [SUPPORTED, INSUFFICIENT, INSUFFICIENT]
    oracle = [SUPPORTED, CONTRADICTED, SUPPORTED]
    stats = decompose(gold, pred, oracle)
    assert stats['coverage'] == pytest.approx(1 / 3)
    assert stats['selective_accuracy'] == 1.0
    assert stats['accuracy'] == pytest.approx(1 / 3)
