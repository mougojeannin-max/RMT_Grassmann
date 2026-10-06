"""Recompute both synthetic graphs, including all five methods, from Gaussian draws."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

from distances import completed_distances
from simulation_metrics import ai_distances, le_distances


SEED = 9236101
DIMENSIONS = tuple(range(32, 257, 32))
METHODS = ("CORR", "NAIVE", "AI-CORR", "LE-CORR", "ORACLE")
GRASSMANN = ("CORR", "NAIVE", "ORACLE")
STYLES = {
    "CORR": ("#0072B2", "o", "-"),
    "NAIVE": ("#CC79A7", "^", "-"),
    "AI-CORR": ("#009E73", "D", "-"),
    "LE-CORR": ("#E69F00", "v", "-"),
    "ORACLE": ("#222222", "s", "--"),
}


def train_size(p):
    return int(np.ceil(.6*np.sqrt(p)))


def test_size(p):
    return int(np.ceil(.4*np.sqrt(p)))


def inverse_sqrt(matrix):
    values, vectors = np.linalg.eigh((matrix+matrix.T)*.5)
    if values[0] <= 0:
        raise ValueError("Whitening requires a positive definite covariance")
    return (vectors*values**-.5)@vectors.T


def populations(p):
    """Four additive rank-one signals over common AR(1) noise."""
    grid = np.arange(1, p+1, dtype=float)[:, None]
    modes = np.arange(1, 5, dtype=float)[None, :]
    basis = np.sqrt(2/(p+1))*np.sin(np.pi*grid*modes/(p+1))
    indices = np.arange(p)
    noise = .3**np.abs(indices[:, None]-indices[None, :])
    theta = np.deg2rad(3.)/2
    signals = []
    for group in range(2):
        first, second = basis[:, 2*group:2*group+1], basis[:, 2*group+1:2*group+2]
        for sign, strength in ((1, 36.), (-1, 45.)):
            direction = np.eye(p)@(np.cos(theta)*first+sign*np.sin(theta)*second)
            signals.append((direction*np.asarray((strength,)))@direction.T)
    covariances = noise[None, :, :]+np.stack(signals)
    return covariances, covariances.mean(axis=0)


def population_reference(p):
    """Known alignments and the population distance between classes 1 and 2."""
    covariances, pool = populations(p)
    transform = inverse_sqrt(pool)
    psi = np.stack([np.linalg.eigvalsh(transform@cov@transform.T)[-1:]
                    for cov in covariances])
    gamma = psi-1.
    if np.any(gamma <= np.sqrt(.1)):
        raise ValueError("The oracle's rank-one signals must be detectable")
    xi = (1.-.1/gamma**2)/(1.+.1/gamma)
    values, vectors = np.linalg.eigh(transform@covariances[:2]@transform.T)
    bases = [v[:, s > 1.+np.sqrt(.1)] for s, v in zip(values, vectors)]
    if any(b.shape[1] != 1 for b in bases):
        raise ValueError("Population reference must have rank one")
    singular = np.linalg.svd(bases[0].T@bases[1], compute_uv=False)
    truth = float(np.linalg.norm(np.arccos(np.clip(singular, 0., 1.))))
    return truth, psi, xi


def generate(p, rng):
    """Generate objects class by class; preserve the paper's RNG draw order."""
    covariances, _ = populations(p)
    factors = np.linalg.cholesky(covariances)
    n = 10*p
    scms, labels, train, test = [], [], [], []
    for label in range(4):
        for count, indices in ((train_size(p), train), (test_size(p), test)):
            for _ in range(count):
                indices.append(len(labels))
                observations = factors[label]@rng.standard_normal((p, n))
                scms.append(observations@observations.T/n)
                labels.append(label)
    train, test = np.asarray(train), np.asarray(test)
    rng.shuffle(train)
    rng.shuffle(test)
    return np.stack(scms), np.asarray(labels), np.full(len(labels), n, dtype=int), train, test


def whiten(scms, sizes, train):
    """Estimate the common pooled covariance using training objects only."""
    weights = sizes[train].astype(float)
    pool = np.einsum("a,aij->ij", weights, scms[train])/weights.sum()
    pool = (pool+pool.T)*.5
    transform = inverse_sqrt(pool)
    white = transform@scms@transform.T
    return (white+white.swapaxes(-1, -2))*.5, int(sizes[train].sum())


def distance_matrices(white, sizes, train, test, n_pool, labels, oracle_xi):
    p = white.shape[1]
    values, vectors = np.linalg.eigh(white)
    corrected_bases, xis, naive_bases = [], [], []
    for spectrum, eigenvectors, n in zip(values, vectors, sizes):
        c = p/n
        threshold = (1.+np.sqrt(c))**2+p**(-.6)+(p/n_pool)**.45
        selected = spectrum > threshold
        corrected_bases.append(eigenvectors[:, selected])
        centered = spectrum[selected]-(1.+c)
        gamma = .5*(centered+np.sqrt(np.maximum(centered**2-4*c, 0.)))
        alignment = np.clip((1.-c/gamma**2)/(1.+c/gamma), np.finfo(float).eps, None)
        xis.append(np.maximum(1/p, alignment))
        naive_bases.append(eigenvectors[:, spectrum > 1.+float(n)**(-1/3)])
    oracle_bases = [v[:, -1:] for v in vectors]
    return {
        "CORR": completed_distances(corrected_bases, test, train, xis=xis),
        "NAIVE": completed_distances(naive_bases, test, train),
        "AI-CORR": ai_distances(white, sizes, test, train),
        "LE-CORR": le_distances(white, sizes, test, train),
        "ORACLE": completed_distances(oracle_bases, test, train, xis=oracle_xi[labels]),
    }


def predict(matrix, labels, keys):
    """Uniform 3-NN; shared random distance ties, nearest majority vote ties."""
    neighbors = np.lexsort((keys, matrix), axis=1)[:, :3]
    predictions = []
    for indices in neighbors:
        selected = labels[indices]
        counts = np.bincount(selected)
        predictions.append(next(int(y) for y in selected if counts[y] == counts.max()))
    return np.asarray(predictions)


def one_repetition(p, repetition, truth, oracle_xi):
    data_seed, tie_seed = np.random.SeedSequence([SEED, p, repetition]).spawn(2)
    scms, labels, sizes, train, test = generate(p, np.random.default_rng(data_seed))
    white, n_pool = whiten(scms, sizes, train)
    keys = np.random.default_rng(tie_seed).random((len(test), len(train)))
    matrices = distance_matrices(white, sizes, train, test, n_pool, labels, oracle_xi)
    pairs = (((labels[test, None] == 0) & (labels[train][None, :] == 1))
             | ((labels[test, None] == 1) & (labels[train][None, :] == 0)))
    if int(pairs.sum()) != 2*train_size(p)*test_size(p):
        raise AssertionError("Incorrect number of class 1 / class 2 pairs")
    accuracy, errors = [], []
    for method, matrix in matrices.items():
        if not np.isfinite(matrix).all():
            raise FloatingPointError("Nonfinite distance matrix: "+method)
        prediction = predict(matrix, labels[train], keys)
        common = dict(p=p, repetition=repetition, seed=SEED, method=method)
        accuracy.append(dict(common, accuracy=float(np.mean(prediction == labels[test])),
                             n_train=len(train), n_test=len(test), n_pool=n_pool))
        if method in GRASSMANN:
            estimates = matrix[pairs].astype(np.float64)
            difference = estimates-truth
            errors.append(dict(common, true_distance=truth,
                mean_distance=float(estimates.mean()), signed_error=float(difference.mean()),
                mae=float(np.abs(difference).mean()), mse=float(np.mean(difference**2)),
                pairs=len(estimates)))
    return dict(accuracy=accuracy, distance=errors)


def worker(job):
    with threadpool_limits(limits=1):
        return one_repetition(*job)


def summarize(frame, metric, half_column):
    summary = frame.groupby(["p", "method"])[metric].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"mean": metric, "std": "sd", "count": "repetitions"})
    summary["se"] = summary.sd/np.sqrt(summary.repetitions)
    summary[half_column] = t.ppf(.975, summary.repetitions-1)*summary.se
    summary["ci95_lower"] = summary[metric]-summary[half_column]
    summary["ci95_upper"] = summary[metric]+summary[half_column]
    return summary


def figures(accuracy, errors, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "serif", "font.size": 8,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "axes.linewidth": .6}):
        for stem, frame, methods, metric, ci, scale, ylabel in (
            ("knn_ar1_sinus_oracle_column", accuracy, METHODS, "accuracy", "half_ci95",
             100., "Classification accuracy (%)"),
            ("grassmann_mae_ar1_sinus_oracle_column", errors, GRASSMANN, "mae", "mae_half_ci95",
             1., "Mean absolute error (rad)"),
        ):
            fig, ax = plt.subplots(figsize=(3.25, 2.55))
            for method in methods:
                group = frame.loc[frame.method == method].sort_values("p")
                color, marker, linestyle = STYLES[method]
                x = group.p.to_numpy()
                y, half = scale*group[metric].to_numpy(), scale*group[ci].to_numpy()
                ax.plot(x, y, label=method, color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.1, markersize=3)
                lower, upper = y-half, y+half
                if metric == "accuracy":
                    lower, upper = np.maximum(lower, 0), np.minimum(upper, 100)
                else:
                    lower = np.maximum(lower, np.finfo(float).tiny)
                ax.fill_between(x, lower, upper, color=color, alpha=.13, linewidth=0)
            if metric == "accuracy":
                ax.axhline(25., color=".5", linestyle=":", linewidth=.8)
                ax.set_ylim(20, 102)
            else:
                ax.set_yscale("log")
            ax.set(xlabel="$p$", ylabel=ylabel, xticks=sorted(frame.p.unique()))
            ax.grid(alpha=.2, linewidth=.5)
            fig.subplots_adjust(left=.19, right=.98, top=.97, bottom=.34)
            ax.tick_params(labelsize=7, pad=2)
            fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
                       bbox_to_anchor=(.55, .005), ncol=3 if len(methods) <= 3 else 2,
                       frameon=False, fontsize=7, handlelength=1.3, columnspacing=.9)
            fig.savefig(output/(stem+".pdf"), metadata={"Creator": "", "Producer": "",
                                                      "CreationDate": None, "ModDate": None})
            fig.savefig(output/(stem+".png"), dpi=220)
            plt.close(fig)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def run(output, dimensions=DIMENSIONS, repetitions=20, workers=1):
    dimensions = list(dimensions)
    if (not dimensions or len(set(dimensions)) != len(dimensions)
            or not set(dimensions) <= set(DIMENSIONS)
            or not 2 <= repetitions <= 20 or workers < 1):
        raise ValueError("Use distinct paper dimensions, 2..20 repetitions and positive workers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    manifest = dict(dimensions=dimensions, repetitions=repetitions, seed=SEED,
        methods=METHODS, distance_methods=GRASSMANN, classes=4, rank=1, c=.1,
        weak=36., strong=45., angle_deg=3., rho=.3, alpha=.6, beta=.45, gamma=1/3,
        neighbors=3, train_sqrt_scale=.6, test_sqrt_scale=.4, sqrt_rounding="ceil",
        protocol="Gaussian observations; additive sinusoidal signals; empirical train-only whitening",
        uncertainty="Pointwise 95% Student intervals across independent repetition means",
        oracle="True rank and population xi; empirical eigenvectors and empirical pooled whitening",
        source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                       for name in ("simulation.py", "simulation_metrics.py", "distances.py")})
    manifest = json.loads(json.dumps(manifest))
    if (output/"config.json").exists():
        previous = json.loads((output/"config.json").read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Output configuration or code changed; use a new output directory")
    write_json(output/"config.json", manifest)
    checkpoint = output/"repetitions"
    checkpoint.mkdir(exist_ok=True)
    jobs, results, protocol, parameters = [], [], [], []
    with threadpool_limits(limits=1):
        for p in dimensions:
            truth, psi, xi = population_reference(p)
            protocol.append(dict(p=p, n=10*p, train_per_class=train_size(p), test_per_class=test_size(p),
                                 repetitions=repetitions, true_distance=truth,
                                 pairs_per_repetition=2*train_size(p)*test_size(p)))
            for label in range(4):
                parameters.append(dict(p=p, class_index=label+1, psi=float(psi[label, 0]),
                                       xi=float(xi[label, 0]), c=.1))
            for repetition in range(repetitions):
                path = checkpoint/f"p_{p}_rep_{repetition:04d}.json"
                if path.exists():
                    results.append(json.loads(path.read_text(encoding="utf-8")))
                else:
                    jobs.append(((p, repetition, truth, xi), path))
    total = len(dimensions)*repetitions
    print(f"{len(results)}/{total} repetitions cached", flush=True)

    def collect(result, path):
        write_json(path, result)
        results.append(result)
        first = result["accuracy"][0]
        print(f'{len(results)}/{total}: p={first["p"]}, repetition={first["repetition"]+1}', flush=True)

    if workers == 1:
        for job, path in jobs:
            collect(worker(job), path)
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker, job): path for job, path in jobs}
            for future in as_completed(futures):
                collect(future.result(), futures[future])
    accuracy_rows = pd.DataFrame([row for result in results for row in result["accuracy"]])
    distance_rows = pd.DataFrame([row for result in results for row in result["distance"]])
    accuracy = summarize(accuracy_rows, "accuracy", "half_ci95")
    errors = summarize(distance_rows, "mae", "mae_half_ci95")
    for name, frame in (("knn_repetitions", accuracy_rows), ("distance_repetitions", distance_rows),
                        ("accuracy_summary", accuracy), ("distance_summary", errors),
                        ("protocol", pd.DataFrame(protocol)), ("oracle_parameters", pd.DataFrame(parameters))):
        keys = [key for key in ("p", "repetition", "method", "class_index") if key in frame]
        frame.sort_values(keys).to_csv(output/(name+".csv"), index=False)
    figures(accuracy, errors, output)
    return accuracy, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/simulation"))
    parser.add_argument("--dimensions", type=int, nargs="+", default=DIMENSIONS)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args.output, args.dimensions, args.repetitions, args.workers)
