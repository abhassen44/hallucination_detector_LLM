"""Log completed inference metrics to workspace-local MLflow; never fit models."""
import hashlib
import json
import os

from src.utils.config import PROJECT_ROOT


def main():
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    import mlflow
    from mlflow.tracking import MlflowClient

    out = PROJECT_ROOT/'experiments/stage5_inference'
    metrics_path = out/'metrics.json'
    raw = metrics_path.read_bytes()
    report = json.loads(raw)
    expected = {'provided_whole_answer','provided_extracted_claims',
                'bm25_extracted_claims','dense_extracted_claims','hybrid_extracted_claims'}
    if set(report['results']) != expected:
        raise ValueError('Wait for all five evidence conditions before logging the final run')
    if report['protocol']['training_performed'] or report['protocol']['threshold_tuning_performed']:
        raise ValueError('This exporter only accepts inference-only results')
    digest = hashlib.sha256(raw).hexdigest()
    db = PROJECT_ROOT/'data/cache/mlflow.db'
    db.parent.mkdir(parents=True,exist_ok=True)
    mlflow.set_tracking_uri(f'sqlite:///{db.as_posix()}')
    client = MlflowClient()
    name = 'hallucination-inference-only'
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        name, artifact_location=(PROJECT_ROOT/'mlartifacts').as_uri())
    runs = []
    for condition, rules in sorted(report['results'].items()):
        for rule, scores in sorted(rules.items()):
            signature = f'{condition}/{rule}'
            previous = client.search_runs([experiment_id], filter_string=(
                f"tags.source_sha256 = '{digest}' AND tags.condition_rule = '{signature}'"))
            if previous:
                run_id = previous[0].info.run_id
            else:
                with mlflow.start_run(experiment_id=experiment_id,run_name=signature) as run:
                    run_id = run.info.run_id
                    mlflow.set_tags({'source_sha256':digest,'condition_rule':signature,
                                     'training_performed':'false','threshold_tuning_performed':'false'})
                    mlflow.log_params({'evidence_condition':condition,'rule':rule,
                        'dataset_sha256':report['dataset_sha256'],
                        'nli_model':report['nli_fingerprint'][0],
                        'extraction_model':report['protocol']['extraction_model'],
                        'entail_threshold':report['protocol']['entail_threshold'],
                        'contra_threshold':report['protocol']['contra_threshold']})
                    mlflow.log_metrics({k:float(v) for k,v in scores.items()})
                    mlflow.log_artifact(str(metrics_path))
                    mlflow.log_artifact(str(out/'protocol.json'))
                    evidence_file = ('provided' if condition.startswith('provided') else condition.split('_')[0])
                    mlflow.log_artifact(str(out/f'predictions_{evidence_file}.jsonl'))
            runs.append({'condition':condition,'rule':rule,'run_id':run_id})
    result={'tracking_uri':f'sqlite:///{db.as_posix()}', 'experiment_id':experiment_id,
            'source_sha256':digest,'training_performed':False,'runs':runs}
    (out/'mlflow_runs.json').write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
