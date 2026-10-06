"""Reproduce Tables 2 and 4 using the frozen train/validation/test splits."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from distances import (alignment_coefficients, completed_distances, configure_distance_threads,
                       pca_distances, rank_threshold)

DATASETS = {'rice': 1, 'corn': 1, 'wheat': 1, 'capgmyo': 18, 'hyser': 20, 'flex': 13}
HSI = ('rice', 'corn', 'wheat')
METHODS = ('Corr_raw', 'Corr_white', 'Naive_raw', 'Naive_white', 'AI_PCA', 'LE_PCA')
SEEDS = [42, 123, 456, 789, 2022, 31415, 27182, 8675309, 104729, 99991]
PARAMS = ('q', 'tau', 'alpha', 'beta', 'gamma')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def input_fingerprints(folder, units):
    """Check frozen object ordering and splits before any expensive computation."""
    hashes = {}
    root = Path(__file__).resolve().parent
    for dataset, subject in units:
        path = Path(folder) / dataset / f'subject{subject:02d}.npz'
        frozen_path = root / 'reference' / 'splits' / f'{dataset}_{subject:02d}.npz'
        with np.load(path, allow_pickle=False) as data, np.load(frozen_path, allow_pickle=False) as frozen:
            for key in frozen.files:
                if key not in data or not np.array_equal(data[key], frozen[key]):
                    raise ValueError(f'Changed frozen metadata: {dataset}/{subject:02d}/{key}')
            separate_covariance = 'covariances' not in data and 'signals' not in data
        paths = [path]
        if separate_covariance:
            paths.append(path.with_name(path.stem + '_covariances.npy'))
        for source in paths:
            hashes[f'{dataset}/{source.name}'] = file_hash(source)
    return hashes


def read_checkpoint(destination, dataset, seed):
    marker = destination / 'complete.json'
    if not marker.exists():
        return None
    hashes = json.loads(marker.read_text(encoding='utf8'))
    for name, expected in hashes.items():
        if Path(name).name != name or file_hash(destination / name) != expected:
            raise ValueError(f'Changed checkpoint output: {dataset}/{seed}/{name}')
    return dataset, seed, pd.read_csv(destination / 'test.csv'), pd.read_csv(destination / 'selected.csv')


def spec(method, q, tau=None, alpha=None, beta=None, gamma=None):
    row = dict(method=method, q=int(q), tau=tau, alpha=alpha, beta=beta, gamma=gamma)
    row['config_id'] = '|'.join([method] + [str(row[k]) for k in PARAMS])
    return row


def entries(methods=METHODS):
    """The complete article grid. Naive has only gamma and q."""
    result = []
    taus, alphas = [i / 10 for i in range(1, 8)], [i / 10 for i in range(1, 7)]
    betas, gammas = [i / 10 for i in range(1, 5)], [i / 10 for i in range(1, 5)]
    for q in (1, 3, 5, 7, 9, 11, 15, 21, 31, 41):
        result += [spec('Corr_raw', q, t, a) for t, a in product(taus, alphas) if t + a < 1 - 1e-12]
        result += [spec('Corr_white', q, t, a, b) for t, a, b in product(taus, alphas, betas)
                   if t + a < 1 - 1e-12]
        result += [spec(m, q, gamma=g) for m in ('Naive_raw', 'Naive_white') for g in gammas]
        result += [spec(m, q) for m in ('AI_PCA', 'LE_PCA')]
    return [entry for entry in result if entry['method'] in methods]


def validate_partition(labels, part, groups=None):
    if part.shape != labels.shape or set(np.unique(part)) != {0, 1, 2}:
        raise ValueError('Three disjoint, exhaustive partitions required')
    classes = set(labels.tolist())
    if any(set(labels[part == i].tolist()) != classes for i in range(3)):
        raise ValueError('Missing class in train/validation/test')
    if groups is not None and len(groups):
        if groups.shape != labels.shape:
            raise ValueError('Invalid group labels')
        sets = [set(groups[part == i].tolist()) for i in range(3)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('Physical group leakage')


def load_data(folder, dataset, subject):
    path = Path(folder) / dataset / f'subject{subject:02d}.npz'
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    required = {'labels', 'dof', 'partitions', 'seeds', 'source_denominator'}
    if required - data.keys():
        raise ValueError(f'Missing fields: {sorted(required - data.keys())}')
    labels = data['labels']
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('One integer class label per object required')
    if data['dof'].shape != labels.shape or np.any(data['dof'] <= 0):
        raise ValueError('Positive degrees of freedom required')
    if not np.array_equal(data['seeds'], SEEDS) or data['partitions'].shape != (len(SEEDS), len(labels)):
        raise ValueError('The ten frozen article partitions are required')
    for partition in data['partitions']:
        validate_partition(labels, partition, data.get('groups'))
    if str(data['source_denominator']) not in ('n', 'n-1'):
        raise ValueError('source_denominator must be n or n-1')
    data['is_emg'] = dataset not in HSI
    if 'signals' in data:
        x = data.pop('signals')
        if (not data['is_emg'] or x.ndim != 3 or x.shape[:2] != (len(labels), 64)
                or not np.isfinite(x).all() or not np.all(data['dof'] == 63)
                or str(data['source_denominator']) != 'n-1'):
            raise ValueError('Expected finite 64-observation EMG windows with dof=63')
        x = x - x.mean(axis=1, keepdims=True)
        data['covariances'] = x.swapaxes(1, 2) @ x / 63
    elif 'covariances' not in data:
        data['covariances'] = np.load(path.with_name(path.stem + '_covariances.npy'),
                                      mmap_mode='r', allow_pickle=False)
    shape = data['covariances'].shape
    if len(shape) != 3 or shape[0] != len(labels) or shape[1] != shape[2] or shape[1] < 2:
        raise ValueError('Expected one square covariance matrix per object')
    return data


def indices(data, seed):
    part = data['partitions'][list(data['seeds']).index(seed)]
    validate_partition(data['labels'], part, data.get('groups'))
    # Fixed, label-independent ordering resolves exact neighbour-distance ties.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 913]))
    return tuple(rng.permutation(np.flatnonzero(part == i)) for i in range(3))


def scope_for(method):
    return 'pca' if method.endswith('_PCA') else method.rsplit('_', 1)[1]


def fit(data, seed, phase, scope):
    """Fit channel selection, whitening and PCA on training objects only."""
    if phase not in ('validation', 'test') or scope not in ('raw', 'white', 'pca'):
        raise ValueError('Unknown phase or preprocessing')
    train, validation, test = indices(data, seed)
    query = validation if phase == 'validation' else test
    ids = np.unique(np.r_[train, query])
    cov = np.array(data['covariances'][ids], dtype=float, copy=True)
    counts = data['dof'][ids]
    if str(data['source_denominator']) == 'n':
        cov *= ((counts + 1) / counts)[:, None, None]
    for start in range(0, len(cov), 8):
        block = cov[start:start + 8]
        cov[start:start + 8] = (block + block.swapaxes(-1, -2)) * .5
    refs, queries = (np.searchsorted(ids, v) for v in (train, query))
    if not np.isfinite(cov).all() or np.any(counts <= 0):
        raise ValueError('Invalid sample covariance matrices')
    keep = np.ones(cov.shape[-1], dtype=bool)
    if data['is_emg']:
        variance = np.diagonal(cov[refs], axis1=1, axis2=2).mean(axis=0)
        if not np.isfinite(variance).all() or variance.max() <= 0:
            raise ValueError('No finite positive training-channel variance')
        keep = variance > variance.max() * 1e-14
        if np.count_nonzero(keep) < 2:
            raise ValueError('Fewer than two active training channels')
        cov = cov[:, keep][:, :, keep]
    p = cov.shape[-1]
    total = int(counts[refs].sum())
    pooled = np.einsum('i,ijk->jk', counts[refs] / total, cov[refs])
    model = dict(train=refs, query=queries, train_ids=train, query_ids=query, pool_ids=train.copy(),
                 indices=ids, dof=counts, p=p, seed=int(seed), phase=phase, scope=scope,
                 n_global=total if scope == 'white' else None, keep=keep, pooled=pooled)
    if scope in ('white', 'pca'):
        ev, vec = np.linalg.eigh((pooled + pooled.T) * .5)
        if ev[0] <= 0 or not np.isfinite(ev).all():
            raise ValueError('Non-SPD training pool; no ridge')
        if scope == 'white':
            white = (vec / np.sqrt(ev)) @ vec.T
            for start in range(0, len(cov), 8):
                cov[start:start + 8] = white @ cov[start:start + 8] @ white.T
            model['whitening'] = white
        else:
            fraction = np.cumsum(ev[::-1]) / ev.sum()
            rank95 = int(np.searchsorted(fraction, .95) + 1)
            # This structural cap uses sample-count metadata, never held-out scores.
            cap = min(p, int(np.min(data['dof'])))
            rank = min(rank95, cap)
            basis = vec[:, ::-1][:, :rank]
            projected = basis.T @ cov @ basis
            model.update(projected=(projected + projected.swapaxes(-1, -2)) * .5,
                         pca_basis=basis, pca_rank=rank, pca_rank95=rank95,
                         pca_cap=cap, pca_fraction=float(fraction[rank - 1]))
            return model
    # Independent small batches avoid several full-size temporary covariance stacks.
    ev = np.empty((len(cov), p))
    for start in range(0, len(cov), 16):
        block = cov[start:start + 16]
        ev[start:start + 16], vec = np.linalg.eigh((block + block.swapaxes(-1, -2)) * .5)
        cov[start:start + 16] = vec
    ev = np.maximum(ev[:, ::-1], 0)
    for i, count in enumerate(counts):
        ev[i, min(p, int(count)):] = 0
    model.update(values=ev, vectors=cov[:, :, ::-1])
    return model


def representations(model, entry):
    values, counts, p = model['values'], model['dof'], model['p']
    xis = None
    if entry['method'].startswith('Naive'):
        if any(entry[k] is not None for k in ('tau', 'alpha', 'beta')):
            raise ValueError('Naive accepts only gamma and q')
        ranks = np.count_nonzero(values > 1 + counts[:, None] ** (-entry['gamma']), axis=1)
    else:
        lp = min(p - 1, max(1, int(np.floor(float(p) ** entry['tau']))))
        noise = values[:, lp:].mean(axis=1)
        if np.any(counts <= lp) or not np.isfinite(noise).all() or np.any(noise <= 0):
            raise ValueError('Noise tail requires floor(p**tau) < n-1 and positive mean')
        thresholds = np.asarray([rank_threshold(p, int(n), model['n_global'],
                     alpha=entry['alpha'], beta=entry['beta'] or 1 / 3) for n in counts])
        ranks = np.count_nonzero(values / noise[:, None] > thresholds[:, None], axis=1)
        xis = [alignment_coefficients(s[:r], p / float(n), v)
               for s, r, n, v in zip(values, ranks, counts, noise)]
    bases = [model['vectors'][i, :, :rank] for i, rank in enumerate(ranks)]
    complements = [model['vectors'][i, :, rank:] for i, rank in enumerate(ranks)]
    return bases, complements, xis, ranks


def predict_all(matrix, labels, neighbors):
    """Majority vote; ties use distance sum, then ascending class label."""
    if not np.isfinite(matrix).all() or any(q < 1 or q > matrix.shape[1] for q in neighbors):
        raise ValueError('Invalid distances or neighbour count')
    order = np.argsort(matrix, axis=1, kind='stable')[:, :max(neighbors)]
    result, classes = [], np.unique(labels)
    for q in neighbors:
        selected = labels[order[:, :q]]
        votes = np.stack([(selected == k).sum(axis=1) for k in classes], axis=1)
        distances = np.take_along_axis(matrix, order[:, :q], axis=1)
        sums = np.stack([np.where(selected == k, distances, 0).sum(axis=1) for k in classes], axis=1)
        sums[votes != votes.max(axis=1, keepdims=True)] = np.inf
        result.append(classes[sums.argmin(axis=1)])
    return np.asarray(result)


def evaluate(data, model, requested, identity):
    rows, predictions, groups = [], [], {}
    labels, truth = data['labels'][model['train_ids']], data['labels'][model['query_ids']]
    for entry in requested:
        if scope_for(entry['method']) != model['scope']:
            raise ValueError('Wrong preprocessing scope')
        key = tuple(entry[k] for k in ('method', 'tau', 'alpha', 'beta', 'gamma'))
        groups.setdefault(key, []).append(entry)
    cache, previous_tau = {}, None
    for candidates in groups.values():
        entry = candidates[0]
        matrix, ranks, error = None, None, ''
        try:
            if model['scope'] == 'pca':
                matrix = pca_distances(model, entry['method'])
            else:
                if entry['tau'] != previous_tau:
                    cache.clear()
                    previous_tau = entry['tau']
                bases, complements, xis, ranks = representations(model, entry)
                signature = (entry['method'], tuple(ranks))
                if signature not in cache:
                    if len(cache) >= 8:
                        cache.pop(next(iter(cache)))
                    cache[signature] = completed_distances(bases, model['query'], model['train'],
                                                           xis=xis, complements=complements if xis is None else None)
                matrix = cache[signature]
            preds = predict_all(matrix, labels, [e['q'] for e in candidates])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            error, matrix = str(exc), None
        for i, entry in enumerate(candidates):
            row = dict(identity, **entry, phase=model['phase'], status='invalid', reason=error,
                       ba=np.nan, accuracy=np.nan, n_train=len(labels), n_query=len(truth), p=model['p'],
                       pca_rank=model.get('pca_rank'), pca_rank95=model.get('pca_rank95'),
                       pca_fraction=model.get('pca_fraction'))
            pred = np.full(len(truth), -1, dtype=np.int64)
            if matrix is not None:
                pred = preds[i]
                row.update(status='ok', reason='', ba=float(balanced_accuracy_score(truth, pred)),
                           accuracy=float(np.mean(truth == pred)))
            if ranks is not None:
                row.update(rank_train_mean=float(ranks[model['train']].mean()),
                           rank_query_mean=float(ranks[model['query']].mean()))
            rows.append(row)
            predictions.append(pred)
    return pd.DataFrame(rows), np.asarray(predictions, dtype=np.int64)


def select(scores, requested, subjects):
    """Select on mean participant validation BA; require every participant."""
    if set(scores.phase) != {'validation'} or scores.dataset.nunique() != 1 or scores.seed.nunique() != 1:
        raise ValueError('Selection requires validation scores for one dataset and split')
    expected = {(s, e['config_id']) for s in subjects for e in requested}
    keys = list(scores[['subject', 'config_id']].itertuples(index=False, name=None))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Incomplete or duplicated validation grid')
    reference = pd.DataFrame(requested).set_index('config_id')
    for subject in subjects:
        part = scores[scores.subject.eq(subject)].set_index('config_id').loc[reference.index]
        if not np.array_equal(part.method, reference.method):
            raise ValueError('Wrong method')
        for key in PARAMS:
            if not np.allclose(part[key].to_numpy(float), reference[key].to_numpy(float),
                               equal_nan=True, rtol=0, atol=1e-12):
                raise ValueError('Changed validation parameters')
    ok = scores.status.eq('ok')
    if set(scores.status) - {'ok', 'invalid'} or not np.isfinite(scores.loc[ok, 'ba']).all() or not scores.loc[ok, 'ba'].between(0, 1).all():
        raise ValueError('Invalid validation scores')
    ranking = []
    for entry in requested:
        part = scores[scores.config_id.eq(entry['config_id'])]
        valid = bool(part.status.eq('ok').all())
        ranking.append(dict(entry, valid=valid, validation_ba=float(part.ba.mean()) if valid else np.nan))
    ranking, chosen = pd.DataFrame(ranking), []
    for method in dict.fromkeys(e['method'] for e in requested):
        valid = ranking[ranking.method.eq(method) & ranking.valid].copy()
        if valid.empty:
            raise ValueError(f'No candidate valid for all participants: {method}')
        valid['rounded'] = valid.validation_ba.round(12)
        winner = valid.sort_values(['rounded', *PARAMS], ascending=[False, True, True, True, True, True],
                                   na_position='first', kind='stable').iloc[0]
        entry = next(e for e in requested if e['config_id'] == winner.config_id)
        chosen.append(dict(entry, validation_ba=float(winner.validation_ba)))
    return chosen


def read_selected(path, dataset, seed, methods):
    table = pd.read_csv(path)
    table = table[table.dataset.eq(dataset) & table.seed.eq(seed) & table.method.isin(methods)]
    if len(table) != len(methods) or set(table.method) != set(methods):
        raise ValueError('Expected exactly one frozen selection per dataset, seed and method')
    allowed = {e['config_id'] for e in entries(methods)}
    selected = []
    for row in table.to_dict('records'):
        entry = spec(row['method'], row['q'], **{k: None if pd.isna(row[k]) else float(row[k])
                                               for k in PARAMS if k != 'q'})
        if entry['config_id'] not in allowed:
            raise ValueError('Frozen selection is outside the article grid')
        if 'config_id' in row and row['config_id'] != entry['config_id']:
            raise ValueError('Frozen selection identity disagrees with its parameters')
        selected.append(dict(entry, validation_ba=row.get('validation_ba', np.nan)))
    return selected


def run_split(task):
    folder, output, dataset, subjects, seed, methods, selected_file, threads, distance_threads = task
    destination = Path(output) / 'runs' / f'{dataset}_{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    requested = entries(methods)
    configure_distance_threads(distance_threads)
    with threadpool_limits(limits=threads):
        if selected_file:
            selected = read_selected(selected_file, dataset, seed, methods)
        else:
            validation = []
            for subject in subjects:
                data = load_data(folder, dataset, subject)
                for scope in ('raw', 'white', 'pca'):
                    current = [e for e in requested if scope_for(e['method']) == scope]
                    if current:
                        model = fit(data, seed, 'validation', scope)
                        scores, _ = evaluate(data, model, current, dict(dataset=dataset, subject=subject, seed=seed))
                        validation.append(scores)
                        del model
                del data
            validation = pd.concat(validation, ignore_index=True)
            validation.to_csv(destination / 'validation.csv.gz', index=False)
            selected = select(validation, requested, subjects)
        selected_table = pd.DataFrame([dict(dataset=dataset, seed=seed, **entry) for entry in selected])
        selected_table.to_csv(destination / 'selected.csv', index=False)
        tests, diagnostics = [], []
        for subject in subjects:
            data = load_data(folder, dataset, subject)
            identity = dict(dataset=dataset, subject=subject, seed=seed)
            for scope in ('raw', 'white', 'pca'):
                current = [e for e in selected if scope_for(e['method']) == scope]
                if not current:
                    continue
                model = fit(data, seed, 'test', scope)
                scores, predictions = evaluate(data, model, current, identity)
                tests.append(scores)
                np.savez_compressed(destination / f'subject{subject:02d}_{scope}_predictions.npz',
                                    predictions=predictions, methods=scores.method.to_numpy(dtype=str),
                                    train_ids=model['train_ids'], query_ids=model['query_ids'],
                                    truth=data['labels'][model['query_ids']])
                diagnostic = dict(identity, scope=scope, fit_ids=model['pool_ids'].tolist(),
                                  train_ids=model['train_ids'].tolist(), query_ids=model['query_ids'].tolist(),
                                  removed_channels=np.flatnonzero(~model['keep']).tolist())
                for key in ('pca_rank', 'pca_rank95', 'pca_cap', 'pca_fraction'):
                    if key in model:
                        diagnostic[key] = model[key]
                diagnostics.append(diagnostic)
                del model
            del data
        tests = pd.concat(tests, ignore_index=True)
        tests.to_csv(destination / 'test.csv', index=False)
        write_json(destination / 'diagnostics.json', diagnostics)
        files = [p for p in destination.iterdir() if p.is_file() and p.name != 'complete.json' and p.suffix != '.tmp']
        write_json(destination / 'complete.json', {p.name: file_hash(p) for p in files})
    return dataset, seed, tests, selected_table


def summarize(test):
    splits = []
    for (dataset, seed, method), frame in test.groupby(['dataset', 'seed', 'method'], sort=True):
        valid = bool(frame.status.eq('ok').all())
        splits.append(dict(dataset=dataset, seed=int(seed), method=method, valid=valid,
                           n_subjects=len(frame), ba=float(frame.ba.mean()) if valid else np.nan))
    splits = pd.DataFrame(splits)
    summary = []
    for (dataset, method), frame in splits.groupby(['dataset', 'method'], sort=True):
        valid = bool(frame.valid.all())
        summary.append(dict(dataset=dataset, method=method, valid=valid, n_splits=len(frame),
                            n_subjects=int(frame.n_subjects.min()),
                            ba_mean=float(frame.ba.mean()) if valid else np.nan,
                            ba_std=float(frame.ba.std(ddof=1)) if valid and len(frame) > 1 else np.nan))
    return splits, pd.DataFrame(summary)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('results/real_data'))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--subjects', nargs='+', type=int, help='Optional participant subset; produces a partial result')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--selected', type=Path, help='Test-only verification using a frozen selection CSV; skips validation')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1, help='BLAS threads per worker')
    parser.add_argument('--distance-threads', type=int, default=1,
                        help='Shared-memory threads for Grassmann matrix rows; default: 1')
    parser.add_argument('--resume', action='store_true', help='Reuse verified completed splits from the same run')
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1 or args.distance_threads < 1:
        parser.error('--workers, --threads and --distance-threads must be positive')
    for name in ('datasets', 'seeds', 'subjects', 'methods'):
        values = getattr(args, name)
        if values is not None and len(values) != len(set(values)):
            parser.error(f'Duplicate --{name}')
    tasks, units = [], []
    for dataset in args.datasets:
        subjects = [0] if dataset in HSI else list(range(1, DATASETS[dataset] + 1))
        if args.subjects:
            subjects = [s for s in subjects if s in args.subjects]
            if not subjects:
                parser.error(f'No requested subject belongs to {dataset}')
        for subject in subjects:
            path = args.data / dataset / f'subject{subject:02d}.npz'
            if not path.is_file():
                parser.error(f'Missing data: {path}. See README.md and prepare_raw.py.')
            units.append((dataset, subject))
        tasks += [(str(args.data), str(args.output), dataset, subjects, seed, args.methods,
                   str(args.selected) if args.selected else None, args.threads, args.distance_threads) for seed in args.seeds]
    args.output.mkdir(parents=True, exist_ok=True)
    exists = any((args.output / name).exists() for name in ('protocol.json', 'runs', 'test.csv'))
    if exists and not args.resume:
        parser.error('Output already contains a run; choose a new --output directory')
    if args.resume and not (args.output / 'protocol.json').is_file():
        parser.error('--resume requires an existing protocol.json')
    protocol = dict(mode='frozen_selection_test_only' if args.selected else 'full_validation_grid',
                    datasets=args.datasets, seeds=args.seeds, methods=args.methods,
                    subjects=args.subjects, workers=args.workers, threads=args.threads,
                    distance_threads=args.distance_threads,
                    partial=bool(args.subjects or set(args.datasets) != set(DATASETS)
                                 or set(args.seeds) != set(SEEDS) or set(args.methods) != set(METHODS)),
                    pca_fit='train', whitening_fit='train', neighbours='train', refit=False,
                    participant_weighting='equal', split_standard_deviation_ddof=1)
    print('Checking frozen metadata and input checksums...', flush=True)
    protocol['input_sha256'] = input_fingerprints(args.data, units)
    code = Path(__file__).resolve().parent
    protocol['code_sha256'] = {name: file_hash(code / name) for name in ('real_data.py', 'distances.py')}
    protocol['versions'] = {name: version(name) for name in ('numpy', 'scipy', 'pandas', 'scikit-learn')}
    protocol['selected_sha256'] = file_hash(args.selected) if args.selected else None
    if args.resume:
        previous = json.loads((args.output / 'protocol.json').read_text(encoding='utf8'))
        resources = {'workers', 'distance_threads'}
        if {k: v for k, v in previous.items() if k not in resources} != {k: v for k, v in protocol.items() if k not in resources}:
            parser.error('Cannot resume: code, data, selections, dependencies or protocol changed')
    write_json(args.output / 'protocol.json', protocol)
    results = []
    if args.resume:
        pending = []
        for task in tasks:
            dataset, seed = task[2], task[4]
            cached = read_checkpoint(args.output / 'runs' / f'{dataset}_{seed}', dataset, seed)
            if cached is None:
                pending.append(task)
            else:
                results.append(cached)
                print(f'Reused {dataset} seed={seed}', flush=True)
        tasks = pending
    if args.workers == 1:
        for task in tasks:
            result = run_split(task)
            results.append(result)
            print(f'Completed {result[0]} seed={result[1]}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_split, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f'Completed {result[0]} seed={result[1]}', flush=True)
    test = pd.concat([r[2] for r in results], ignore_index=True).sort_values(['dataset', 'seed', 'subject', 'method'])
    selected = pd.concat([r[3] for r in results], ignore_index=True).sort_values(['dataset', 'seed', 'method'])
    splits, summary = summarize(test)
    for name, frame in [('test', test), ('selected', selected), ('split_scores', splits), ('summary', summary)]:
        frame.to_csv(args.output / f'{name}.csv', index=False)
    for number, datasets in ((2, HSI), (4, [d for d in DATASETS if d not in HSI])):
        table = summary[summary.dataset.isin(datasets)].copy()
        table['BA (%)'] = [f'{100 * r.ba_mean:.2f} +/- {100 * r.ba_std:.2f}' if r.valid else 'invalid'
                          for r in table.itertuples()]
        table.pivot(index='method', columns='dataset', values='BA (%)').reindex(METHODS).to_csv(args.output / f'table{number}.csv')
    if not test.status.eq('ok').all():
        raise SystemExit('Some methods were invalid; inspect test.csv. No participant was omitted.')
    print(f'Saved {len(test)} test results in {args.output}', flush=True)


if __name__ == '__main__':
    main()
