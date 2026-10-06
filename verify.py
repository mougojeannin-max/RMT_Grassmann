"""Compare recomputed outputs with the frozen numerical paper references."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from references import reference_file


def array_checksum(array):
    digest = hashlib.sha256()
    for start in range(0, len(array), 8):
        digest.update(np.ascontiguousarray(array[start:start+8]).tobytes())
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=digest.hexdigest())


def data_files(folder):
    expected = json.load(reference_file("data_checksums.json"))
    for unit, checksum in expected.items():
        dataset, subject = unit.rsplit("_", 1)
        path = folder / dataset / ("subject" + subject + ".npz")
        with np.load(path, allow_pickle=False) as archive, np.load(
                reference_file("splits/" + unit + ".npz"), allow_pickle=False) as frozen:
            for field in frozen.files:
                np.testing.assert_array_equal(archive[field], frozen[field], err_msg=f"{unit}: {field}")
            if "signals" in archive:
                array = archive["signals"]
            elif "covariances" in archive:
                array = archive["covariances"]
            else:
                array = np.load(path.with_name(path.stem + "_covariances.npy"), mmap_mode="r")
            if array_checksum(array) != checksum:
                raise ValueError(f"{unit}: data checksum differs from the prepared paper inputs")
    return dict(units=len(expected), metadata="identical", data="identical SHA256")


def compare(actual, expected, keys, values, partial=False, atol=1e-12, rtol=0):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("Duplicate experimental result")
    actual, expected = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    missing = expected.index.difference(actual.index)
    extra = actual.index.difference(expected.index)
    if len(extra) or (len(missing) and not partial) or actual.empty:
        raise ValueError(f"Result coverage differs: {len(missing)} missing, {len(extra)} unexpected")
    target = expected.loc[actual.index, values].to_numpy(float)
    computed = actual[values].to_numpy(float)
    if not np.isfinite(computed).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite result")
    np.testing.assert_allclose(computed, target, rtol=rtol, atol=atol)
    return dict(rows=len(actual), missing_rows=len(missing), complete=not len(missing),
                maximum_absolute_error=float(np.max(np.abs(computed-target))),
                absolute_tolerance=atol, relative_tolerance=rtol)


def figure1(folder):
    actual = np.loadtxt(folder / "figure1_histogram.csv", delimiter=",", skiprows=1)
    expected = np.loadtxt(reference_file("figure1_histogram.csv"), delimiter=",", skiprows=1)
    # The reference was independently measured from the published vector PDF.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-6)
    values = np.loadtxt(folder / "figure1_eigenvalues.csv", delimiter=",", skiprows=1)
    edge = (1 + np.sqrt(256/396))**2
    if (values.shape != (256,) or not np.isfinite(values).all() or np.any(values < 0)
            or int(np.sum(values > edge)) != 2):
        raise ValueError("Unexpected RICE spectrum")
    heights, edges = np.histogram(values, bins=35, density=True)
    np.testing.assert_allclose(np.c_[edges[:-1], np.diff(edges), heights], actual, rtol=0, atol=1e-12)
    return dict(bins=len(actual), eigenvalues=len(values), upper_outliers=2,
                maximum_histogram_error=float(np.max(np.abs(actual-expected))))


def simulation(folder, partial):
    result = {}
    for name, expected_name, fields, tolerance in (
        ("knn_repetitions", "simulation_accuracy", ["accuracy", "n_train", "n_test", "n_pool"], 1e-12),
        ("distance_repetitions", "simulation_mae", ["true_distance", "mean_distance", "mae", "mse", "pairs"], 1e-7),
    ):
        expected = pd.read_csv(reference_file(expected_name + ".csv"))
        actual = pd.read_csv(folder / (name + ".csv"))
        train_count = np.ceil(.6*np.sqrt(actual.p)).astype(int)
        test_count = np.ceil(.4*np.sqrt(actual.p)).astype(int)
        if name == "knn_repetitions":
            np.testing.assert_array_equal(actual.n_train, 4*train_count)
            np.testing.assert_array_equal(actual.n_test, 4*test_count)
            np.testing.assert_array_equal(actual.n_pool, 4*train_count*10*actual.p)
        else:
            np.testing.assert_array_equal(actual.pairs, 2*train_count*test_count)
        result[name] = compare(actual, expected,
                               ["p", "repetition", "method"], fields, partial, tolerance)
        key = ["p", "repetition", "method"]
        target = expected.set_index(key).loc[actual.set_index(key).index].reset_index()
        metric, summary_file, half = (("accuracy", "accuracy_summary.csv", "half_ci95")
                if name == "knn_repetitions" else ("mae", "distance_summary.csv", "mae_half_ci95"))
        summary = target.groupby(["p", "method"])[metric].agg(["mean", "std", "count"]).reset_index()
        summary = summary.rename(columns={"mean": metric, "std": "sd", "count": "repetitions"})
        summary["se"] = summary.sd / np.sqrt(summary.repetitions)
        summary[half] = t.ppf(.975, summary.repetitions-1) * summary.se
        summary["ci95_lower"], summary["ci95_upper"] = summary[metric]-summary[half], summary[metric]+summary[half]
        result[summary_file] = compare(pd.read_csv(folder / summary_file), summary,
                ["p", "method"], [metric, "sd", "repetitions", "se", half, "ci95_lower", "ci95_upper"],
                atol=tolerance)
    return result


def tables(folder, partial):
    test = pd.read_csv(folder / "test.csv")
    if not test.status.eq("ok").all() or set(test.phase) != {"test"}:
        raise ValueError("Invalid or non-test rows")
    result = {"test_scores": compare(test, pd.read_csv(reference_file("scores.csv")),
              ["dataset", "subject", "seed", "method"], ["ba", "accuracy"], partial)}
    keys = ["dataset", "subject", "seed", "method"]
    target = pd.read_csv(reference_file("scores.csv")).set_index(keys).loc[test.set_index(keys).index]
    fields = ["q", "tau", "alpha", "beta", "gamma"]
    np.testing.assert_allclose(test[fields].to_numpy(float), target[fields].to_numpy(float),
                               rtol=0, atol=1e-12, equal_nan=True)
    if result["test_scores"]["complete"]:
        # Aggregate independently: equally weighted participants first, then splits.
        split = test.groupby(["dataset", "method", "seed"]).ba.mean()
        summary = split.groupby(["dataset", "method"]).agg(["mean", "std"]).reset_index()
        summary = summary.rename(columns={"mean": "ba_mean", "std": "ba_std"})
        result["published_tables"] = compare(summary, pd.read_csv(reference_file("tables.csv")),
                ["dataset", "method"], ["ba_mean", "ba_std"])
        result["saved_summary"] = compare(pd.read_csv(folder / "summary.csv"), summary,
                ["dataset", "method"], ["ba_mean", "ba_std"])
    else:
        result["published_tables"] = "Not verified: incomplete participants, splits or methods"
    return result


def selection(folder, partial):
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf8"))
    if protocol["mode"] != "full_validation_grid":
        raise ValueError("Selection verification requires a freshly run full validation grid")
    actual = pd.read_csv(folder / "selected.csv")
    expected = pd.read_csv(reference_file("selected.csv"))
    keys = ["dataset", "seed", "method"]
    result = compare(actual, expected, keys, ["validation_ba"], partial)
    expected = expected.set_index(keys).loc[actual.set_index(keys).index]
    for field in ("q", "tau", "alpha", "beta", "gamma"):
        np.testing.assert_allclose(actual[field].to_numpy(float), expected[field].to_numpy(float),
                                   rtol=0, atol=1e-12, equal_nan=True)
    if protocol.get("subjects"):
        raise ValueError("A participant subset cannot verify dataset-wide hyperparameter selection")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure1", type=Path)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--data", type=Path, help="Check all 54 prepared units and frozen partitions")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Check a smoke run; missing results remain explicitly unverified")
    parser.add_argument("--output", type=Path, help="Optional machine-readable verification report")
    args = parser.parse_args()
    if not any((args.figure1, args.simulation, args.tables, args.selection, args.data)):
        args.figure1, args.simulation, args.tables = [Path("results") / p
                                                     for p in ("figure1", "simulation", "real_data")]
    report = {}
    if args.data:
        report["data"] = data_files(args.data)
    if args.figure1:
        report["figure1"] = figure1(args.figure1)
    if args.simulation:
        report["simulation"] = simulation(args.simulation, args.allow_partial)
    if args.tables:
        report["tables"] = tables(args.tables, args.allow_partial)
    if args.selection:
        report["validation_selection"] = selection(args.selection, args.allow_partial)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf8")
    print(text)


if __name__ == "__main__":
    main()
