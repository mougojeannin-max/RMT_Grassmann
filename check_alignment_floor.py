"""Audit whether replacing the machine-epsilon floor by 1/p affects the tables.

For a retained normalized eigenvalue y, y > (1+sqrt(c))**2 + margin.
Above the MP edge, t(y) = (y-1-c + sqrt((y-1-c)**2-4*c))/2 increases,
and chi(t) = (1-c/t**2)/(1+c/t) increases for t > sqrt(c).
Evaluating chi at the smallest selection threshold therefore bounds every
retained coefficient, for every noise estimate and every point in the grid.

This is an exhaustive equivalence check for the clipping change. It does not
rerun the distance matrices or the hyperparameter search. The frozen scores
are aggregated only after this equivalence has been established.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from distances import alignment_coefficients, rank_threshold
import real_data as real
from references import reference_file
from verify import compare, data_files


def old_alignment(values, c, noise):
    centered = np.asarray(values) - (1 + c) * noise
    spikes = .5 * (centered + np.sqrt(np.maximum(centered ** 2 - 4 * c * noise ** 2, 0)))
    xi = (1 - c * noise ** 2 / spikes ** 2) / (1 + c * noise / spikes)
    return np.clip(xi, np.finfo(float).eps, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('results/alignment_floor'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print('Checking all prepared inputs against their frozen checksums...', flush=True)
    inputs = data_files(args.data)
    scores = pd.read_csv(reference_file('scores.csv'))
    grid = real.entries(['Corr_raw', 'Corr_white'])
    maxima = {method: max(e['alpha'] for e in grid if e['method'] == method)
              for method in ('Corr_raw', 'Corr_white')}
    beta_max = max(e['beta'] for e in grid if e['method'] == 'Corr_white')
    records = []
    dimension_checks = 0
    with threadpool_limits(limits=1):
        for dataset, subjects in real.DATASETS.items():
            for subject in ([0] if dataset in real.HSI else range(1, subjects + 1)):
                data = real.load_data(args.data, dataset, subject)
                diagonal = np.diagonal(data['covariances'], axis1=1, axis2=2)
                for seed in real.SEEDS:
                    train = real.indices(data, seed)[0]
                    p = diagonal.shape[1]
                    if data['is_emg']:
                        # EMG inputs use n-1; this matches real.fit's channel filter.
                        assert str(data['source_denominator']) == 'n-1'
                        variance = diagonal[train].mean(axis=0)
                        p = int(np.count_nonzero(variance > variance.max() * 1e-14))
                    actual = scores[(scores.dataset == dataset) & (scores.subject == subject)
                                    & (scores.seed == seed)]
                    assert len(actual) == len(real.METHODS)
                    assert (actual.p == p).all()
                    dimension_checks += len(actual)
                    n_global = int(data['dof'][train].sum())
                    # p/n_global < 1 makes beta_max the smallest whitening margin.
                    assert 0 < p / n_global < 1
                    for method in ('Corr_raw', 'Corr_white'):
                        for n in np.unique(data['dof']):
                            threshold = rank_threshold(p, int(n),
                                n_global if method == 'Corr_white' else None,
                                alpha=maxima[method], beta=beta_max)
                            c = p / float(n)
                            before = old_alignment([threshold], c, 1.)
                            after = alignment_coefficients([threshold], c, 1., p)
                            assert before[0] > 1 / p
                            np.testing.assert_array_equal(before, after)
                            records.append(dict(dataset=dataset, subject=subject, seed=seed,
                                method=method, p=p, n=int(n), n_global=n_global,
                                normalized_threshold=float(threshold),
                                chi_lower_bound=float(before[0]), floor=1 / p,
                                ratio_to_floor=float(before[0] * p)))
                del data, diagonal
            print(f'Grid floor checked: {dataset}', flush=True)
    bounds = pd.DataFrame(records)
    bounds.to_csv(args.output / 'grid_bounds.csv', index=False)
    split_scores, tables = real.summarize(scores)
    table_check = compare(tables, pd.read_csv(reference_file('tables.csv')),
                          ['dataset', 'method'], ['ba_mean', 'ba_std'])
    split_scores.to_csv(args.output / 'unchanged_split_scores.csv', index=False)
    tables.to_csv(args.output / 'unchanged_tables.csv', index=False)
    for suffix, datasets in [('2a', real.HSI), ('2b', ('capgmyo', 'hyser', 'flex'))]:
        frame = tables[tables.dataset.isin(datasets)].copy()
        frame['BA (%)'] = [f'{100*r.ba_mean:.2f} +/- {100*r.ba_std:.2f}'
                          for r in frame.itertuples()]
        frame.pivot(index='method', columns='dataset', values='BA (%)').reindex(real.METHODS).to_csv(
            args.output / f'unchanged_table{suffix}.csv')
    report = dict(
        verification='analytical_equivalence_of_clipping_over_full_grid',
        distance_matrices_recomputed=False, hyperparameter_search_recomputed=False,
        inputs=inputs, grid_entries=len(grid), checked_dataset_subject_splits=dimension_checks // 6,
        checked_published_score_dimensions=dimension_checks, threshold_checks=len(bounds),
        clipping_can_activate=False,
        smallest_alignment_lower_bound=bounds.loc[bounds.chi_lower_bound.idxmin()].to_dict(),
        smallest_ratio_to_floor=bounds.loc[bounds.ratio_to_floor.idxmin()].to_dict(),
        dataset_bounds=bounds.groupby('dataset').agg(
            min_chi=('chi_lower_bound', 'min'), max_floor=('floor', 'max'),
            min_ratio=('ratio_to_floor', 'min')).reset_index().to_dict('records'),
        corrected_scores_proven_unchanged=int(scores.method.str.startswith('Corr').sum()),
        untouched_baseline_scores=int((~scores.method.str.startswith('Corr')).sum()),
        aggregate_comparison=table_check,
        code_sha256={name: real.file_hash(Path(__file__).resolve().parent / name)
                     for name in ('distances.py', 'real_data.py', 'reference.zip')})
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
