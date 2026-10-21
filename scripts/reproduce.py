"""Reproduce the pipeline stage by stage, from one entry point.

    python -m scripts.reproduce --stage all --dry-run
    python -m scripts.reproduce --stage learned

Stages are ordered by dependency. Long GPU stages are never started implicitly by
``--stage all`` unless ``--include-heavy`` is passed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from src.utils.config import PROJECT_ROOT

STAGES = {
    'datasets': [
        ('python', '-m', 'src.datasets.prepare', '--dataset', 'all'),
    ],
    'oracle': [
        ('python', '-m', 'src.evaluation.oracle_nli'),
    ],
    'retrieval': [
        ('python', '-m', 'src.evaluation.fever_retrieval'),
    ],
    'claims': [
        ('python', '-m', 'src.generation.claim_extraction', '--split', 'all', '--dataset', 'halueval'),
        ('python', '-m', 'src.evaluation.claim_extraction_eval', '--split', 'all'),
    ],
    'inference': [
        ('python', '-m', 'src.evaluation.halueval_claims'),
        ('python', '-m', 'scripts.log_inference_mlflow'),
        ('python', '-m', 'scripts.summarize_inference'),
    ],
    'learned': [
        ('python', '-m', 'src.evaluation.fit_detector', '--method', 'all'),
        ('python', '-m', 'scripts.summarize_stage5'),
    ],
    'studies': [
        ('python', '-m', 'src.evaluation.val_studies', '--study', 'all'),
    ],
    'errors': [
        ('python', '-m', 'scripts.analyze_errors', '--queue', 'all'),
    ],
    'checks': [
        ('python', '-m', 'pytest', '-q'),
        ('npm', 'ci', '--prefix', 'frontend'),
        ('npm', 'run', 'build', '--prefix', 'frontend'),
    ],
}
HEAVY = {'datasets', 'oracle', 'retrieval', 'claims', 'inference', 'learned', 'studies'}


def _command(parts: tuple[str, ...]) -> list[str]:
    if parts[0] == 'python':
        return [sys.executable, *parts[1:]]
    if parts[0] == 'npm':
        return ['npm.cmd' if os.name == 'nt' else 'npm', *parts[1:]]
    raise ValueError(f'Unknown runner: {parts[0]}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=[*STAGES, 'all'], default='all')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--include-heavy', action='store_true',
                        help='With --stage all, also run GPU and download stages.')
    parser.add_argument('--keep-going', action='store_true')
    args = parser.parse_args()

    names = list(STAGES) if args.stage == 'all' else [args.stage]
    if args.stage == 'all' and not args.include_heavy:
        names = [name for name in names if name not in HEAVY]
        print('Skipping heavy stages (pass --include-heavy):',
              ', '.join(sorted(HEAVY & set(STAGES))) or 'none')

    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT), PYTHONUTF8='1')
    for name in names:
        for parts in STAGES[name]:
            command = _command(parts)
            print(f'[{name}] {" ".join(parts)}', flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
            if result.returncode:
                print(f'[{name}] failed with exit code {result.returncode}', flush=True)
                if not args.keep_going:
                    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
