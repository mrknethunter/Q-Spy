"""Validation-only nu/tolerance feasibility diagnostic; never uses test rows.

This is a post-review diagnostic, not a change to the production E-03 fit.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.splits import verify_dataset
from experiments.e03_novelty import _hash, _id_hash, _json
from features.transforms import make_features
from models.baselines import build_baseline


def diagnose(directory, curated_csv):
    directory = Path(directory)
    audit = next(row for row in json.loads((directory / 'audit.json').read_text())
                 if row['held_out'] == 'Fast_Scattering' and row['seed'] == 0)
    with gzip.open(directory / 'selected_ids.json.gz', 'rt') as file:
        ids = next(row for row in json.load(file)
                   if row['held_out'] == 'Fast_Scattering' and row['seed'] == 0)
    frame = pd.read_csv(curated_csv, float_precision='round_trip')
    verify_dataset(frame)
    indexed = frame.set_index('id', drop=False)
    train, validation = indexed.loc[ids['train']], indexed.loc[ids['val']]
    assert not train['label'].isin(audit['unknown_classes']).any()
    assert not validation['label'].isin(audit['unknown_classes']).any()
    X, scaler = make_features(train)
    V, _ = make_features(validation, scaler)
    gamma = audit['models']['rbf_ocsvm']['best_params']['gamma']
    records = []
    model = build_baseline('rbf_ocsvm', seed=0)
    for tol in (.001, 1e-6):
        for nu in (1e-5, 1e-4, .001, .005, .01, .05, .1, .2, .5, .9):
            estimator = model._build_estimator(dict(gamma=gamma, nu=nu)).set_params(tol=tol).fit(X)
            records.append(dict(tol=tol, nu=nu, native_known_fpr=float(np.mean(estimator.predict(V) == -1))))
    return dict(scope='post_review_known_validation_only_diagnostic',
                case=dict(held_out='Fast_Scattering', seed=0, model='rbf_ocsvm'),
                no_test_rows_used=True, no_production_parameters_or_results_changed=True,
                train_id_sha256=_id_hash(ids['train']), validation_id_sha256=_id_hash(ids['val']),
                gamma=gamma, records=records, minimum_observed_fpr=min(row['native_known_fpr'] for row in records),
                script_sha256=_hash(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--curated-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(_json(diagnose(args.directory, args.curated_csv)), encoding='utf-8')
