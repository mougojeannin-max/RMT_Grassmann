"""Scientific regression, data isolation and independent reference checks."""
import copy
import io
from pathlib import Path
import sys
import tempfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from distances import completed_distances, configure_distance_threads, pca_distances
import real_data as real
from references import reference_file
from simulation import SEED, generate, population_reference, whiten, worker
from verify import compare, figure1


def fixture():
    rng = np.random.default_rng(193)
    labels = np.repeat(np.arange(3), 30)
    x = rng.normal(size=(90, 24, 6))
    x[:, :, 0] *= 3
    x[np.arange(90), :, labels + 1] *= 2
    x -= x.mean(axis=1, keepdims=True)
    partition = np.tile(np.repeat([0, 1, 2], [18, 6, 6]), 3)
    return dict(labels=labels, covariances=x.swapaxes(1, 2) @ x / 23,
                dof=np.full(90, 23), partitions=np.stack([partition] * 10),
                seeds=np.asarray(real.SEEDS), source_denominator=np.asarray('n-1'),
                groups=np.asarray([], dtype=str), is_emg=False)


class ScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_grassmann_matches_pairwise_svd_and_rank_completion(self):
        rng = np.random.default_rng(61)
        bases, complements, xis = [], [], []
        for rank in (0, 1, 4, 7, 8):
            frame = np.linalg.qr(rng.normal(size=(8, 8)))[0]
            bases.append(frame[:, :rank])
            complements.append(frame[:, rank:])
            xis.append(rng.uniform(.3, 1, rank))
        for correction in (None, xis):
            matrix = completed_distances(bases, np.arange(5), np.arange(5),
                                         xis=correction, complements=complements)
            for i, u in enumerate(bases):
                for j, v in enumerate(bases):
                    gram = u.T @ v
                    if correction is not None:
                        gram = gram / np.sqrt(xis[i][:, None] * xis[j][None, :])
                    singular = np.linalg.svd(gram, compute_uv=False)
                    expected = np.sqrt(np.sum(np.arccos(np.clip(singular, 0, 1)) ** 2)
                                       + np.pi ** 2 / 4 * abs(u.shape[1] - v.shape[1]))
                    self.assertAlmostEqual(float(matrix[i, j]), expected, places=5)

    def test_parallel_rows_preserve_every_distance_bit(self):
        rng = np.random.default_rng(44)
        ranks = [64] * 32 + [0, 40, 64, 96]
        frames = [np.linalg.qr(rng.normal(size=(96, 96)))[0] for _ in ranks]
        bases = [frame[:, :rank] for frame, rank in zip(frames, ranks)]
        complements = [frame[:, rank:] for frame, rank in zip(frames, ranks)]
        xis = [rng.uniform(.4, .9, rank) for rank in ranks]
        query, train = np.arange(32, 36), np.arange(32)
        try:
            for coefficients in (None, xis):
                configure_distance_threads(1)
                expected = completed_distances(bases, query, train, coefficients, complements)
                for workers in (2, 4):
                    configure_distance_threads(workers)
                    actual = completed_distances(bases, query, train, coefficients, complements)
                    np.testing.assert_array_equal(actual, expected)
        finally:
            configure_distance_threads(1)

    def test_pca_distances_match_diagonal_closed_form(self):
        diagonals = np.array([[1., 2., 4.], [3., 5., 2.], [7., 1., 8.]])
        model = dict(projected=np.array([np.diag(x) for x in diagonals]),
                     train=np.array([0, 1]), query=np.array([2]), pca_rank=3)
        expected = np.sqrt(np.mean(np.log(diagonals[2] / diagonals[:2]) ** 2, axis=1))[None]
        for method in ('AI_PCA', 'LE_PCA'):
            np.testing.assert_allclose(pca_distances(model, method), expected, atol=1e-13)

    def test_pca_rejects_singular_covariances_without_regularization(self):
        model = dict(projected=np.array([np.eye(2), np.diag([1., 0.])]),
                     train=np.array([0]), query=np.array([1]), pca_rank=2)
        for method in ('AI_PCA', 'LE_PCA'):
            with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                pca_distances(model, method)

    def test_naive_uses_absolute_strict_threshold_without_noise_estimation(self):
        gamma, n = .2, 19
        threshold = 1 + n ** (-gamma)
        model = dict(p=5, values=np.array([[3., threshold, np.nextafter(threshold, np.inf), .01, 0.]]),
                     vectors=np.eye(5)[None], dof=np.array([n]))
        entry = real.spec('Naive_raw', 1, gamma=gamma)
        with patch.object(real, 'alignment_coefficients', side_effect=AssertionError('No correction')):
            self.assertEqual(real.representations(model, entry)[3][0], 2)
            scaled = dict(model, values=model['values'] * .01)
            self.assertEqual(real.representations(scaled, entry)[3][0], 0)
        naive = real.entries(['Naive_raw', 'Naive_white'])
        self.assertEqual(len(naive), 80)
        self.assertTrue(all(e['tau'] is None and e['alpha'] is None and e['beta'] is None for e in naive))

    def test_held_out_values_and_labels_do_not_change_fitted_transform(self):
        data = fixture()
        train, validation, test = real.indices(data, 42)
        changed = copy.deepcopy(data)
        changed['covariances'][np.r_[validation, test]] *= 1e7
        changed['labels'][np.r_[validation, test]] = (changed['labels'][np.r_[validation, test]] + 1) % 3
        for scope in ('white', 'pca'):
            before = real.fit(data, 42, 'test', scope)
            after = real.fit(changed, 42, 'test', scope)
            np.testing.assert_array_equal(before['pool_ids'], train)
            np.testing.assert_array_equal(before['train_ids'], train)
            self.assertFalse(set(before['pool_ids']) & set(np.r_[validation, test]))
            for key in ('pooled', 'keep', 'whitening' if scope == 'white' else 'pca_basis'):
                np.testing.assert_array_equal(before[key], after[key])
            validation_fit = real.fit(data, 42, 'validation', scope)
            np.testing.assert_array_equal(before['pooled'], validation_fit['pooled'])

    def test_channel_filter_is_train_only_and_pca_cap_uses_sample_counts(self):
        data = fixture()
        data['is_emg'] = True
        train, validation, test = real.indices(data, 42)
        data['covariances'][train, -1, :] = 0
        data['covariances'][train, :, -1] = 0
        data['covariances'][np.r_[validation, test], -1, -1] = 1e12
        model = real.fit(data, 42, 'test', 'pca')
        self.assertEqual(model['p'], 5)
        self.assertFalse(model['keep'][-1])
        self.assertLessEqual(model['pca_rank'], min(5, data['dof'].min()))

    def test_selection_uses_equal_participant_weight_and_validation_only(self):
        requested = [real.spec('Naive_raw', q, gamma=.2) for q in (1, 3)]
        scores = []
        for subject in (1, 2):
            for entry in requested:
                ba = (.9 if subject == 1 else .1) if entry['q'] == 1 else (.5 if subject == 1 else .6)
                scores.append(dict(entry, subject=subject, dataset='capgmyo', seed=42,
                                   phase='validation', n_query=10000 if subject == 1 else 10,
                                   status='ok', ba=ba))
        frame = pd.DataFrame(scores)
        self.assertEqual(real.select(frame, requested, [1, 2])[0]['q'], 3)
        with self.assertRaises(ValueError):
            real.select(frame.assign(phase='test'), requested, [1, 2])
        with self.assertRaises(ValueError):
            real.select(frame.iloc[:-1], requested, [1, 2])
        with self.assertRaises(ValueError):
            real.select(frame.assign(status='invalid', ba=np.nan), requested, [1, 2])

    def test_group_leakage_is_rejected(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        part = np.repeat([0, 1, 2], 2)
        with self.assertRaisesRegex(ValueError, 'group leakage'):
            real.validate_partition(labels, part, np.array(['a', 'b', 'a', 'c', 'd', 'e']))

    def test_report_averages_participants_before_computing_split_sd(self):
        rows = [dict(dataset='capgmyo', seed=seed, subject=subject, method='Corr_raw',
                     status='ok', ba=ba, n_query=10000 if subject == 1 else 10)
                for seed, values in ((42, (.2, .8)), (123, (.6, .8)))
                for subject, ba in enumerate(values, 1)]
        splits, summary = real.summarize(pd.DataFrame(rows))
        np.testing.assert_allclose(splits.ba, [.5, .7])
        self.assertAlmostEqual(summary.ba_mean.iloc[0], .6)
        self.assertAlmostEqual(summary.ba_std.iloc[0], np.std([.5, .7], ddof=1))
        rows[0]['status'] = 'invalid'
        _, summary = real.summarize(pd.DataFrame(rows))
        self.assertFalse(summary.valid.iloc[0])
        self.assertTrue(np.isnan(summary.ba_mean.iloc[0]))

    def test_all_six_methods_evaluate_and_neighbors_are_training_only(self):
        data = fixture()
        train, _, test = real.indices(data, 42)
        for scope in ('raw', 'white', 'pca'):
            model = real.fit(data, 42, 'test', scope)
            requested = [e for e in real.entries() if real.scope_for(e['method']) == scope and e['q'] == 1]
            requested = [next(e for e in requested if e['method'] == m)
                         for m in real.METHODS if real.scope_for(m) == scope]
            rows, predictions = real.evaluate(data, model, requested, dict(dataset='rice', subject=0, seed=42))
            self.assertTrue(rows.status.eq('ok').all())
            self.assertEqual(predictions.shape, (2, len(test)))
            self.assertTrue(rows.n_train.eq(len(train)).all())

    def test_dataset_loader_converts_emg_signals_to_centered_scm(self):
        data = fixture()
        rng = np.random.default_rng(812)
        signals = rng.normal(size=(90, 64, 6)) + 123
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'capgmyo'
            folder.mkdir()
            metadata = {k: v for k, v in data.items() if k not in ('covariances', 'is_emg')}
            metadata['dof'] = np.full(90, 63)
            np.savez_compressed(folder / 'subject01.npz', **metadata, signals=signals)
            loaded = real.load_data(directory, 'capgmyo', 1)
        expected = np.stack([np.cov(x, rowvar=False, ddof=1) for x in signals])
        np.testing.assert_allclose(loaded['covariances'], expected, atol=1e-13)

    def test_frozen_metadata_and_data_fingerprints_detect_changes(self):
        data = {k: v for k, v in fixture().items() if k != 'is_emg'}
        metadata = {k: v for k, v in data.items() if k != 'covariances'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/rice').mkdir(parents=True)
            path = root / 'data/rice/subject00.npz'
            packed = io.BytesIO()
            np.savez_compressed(packed, **metadata)
            with ZipFile(root / 'reference.zip', 'w') as archive:
                archive.writestr('splits/rice_00.npz', packed.getvalue())
            np.savez_compressed(path, **data)
            with patch.object(real, '__file__', str(root / 'real_data.py')):
                before = real.input_fingerprints(root / 'data', [('rice', 0)])
                data['covariances'] *= 2
                np.savez_compressed(path, **data)
                self.assertNotEqual(before, real.input_fingerprints(root / 'data', [('rice', 0)]))
                data['partitions'][0, 0] = 1
                np.savez_compressed(path, **data)
                with self.assertRaisesRegex(ValueError, 'Changed frozen metadata'):
                    real.input_fingerprints(root / 'data', [('rice', 0)])

    def test_resume_rejects_modified_checkpoint_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pd.DataFrame([dict(method='Corr_raw', ba=.5)]).to_csv(folder / 'test.csv', index=False)
            pd.DataFrame([real.spec('Corr_raw', 1, .2, .3)]).to_csv(folder / 'selected.csv', index=False)
            real.write_json(folder / 'complete.json', {name: real.file_hash(folder / name)
                                                       for name in ('test.csv', 'selected.csv')})
            result = real.read_checkpoint(folder, 'rice', 42)
            self.assertEqual(result[2].ba.iloc[0], .5)
            pd.DataFrame([dict(method='Corr_raw', ba=1.)]).to_csv(folder / 'test.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Changed checkpoint output'):
                real.read_checkpoint(folder, 'rice', 42)


class SimulationTests(unittest.TestCase):
    def test_paper_repetitions(self):
        # Independently recomputed with the original experimental kernels.
        accuracies = (
            {"CORR": 5/12, "NAIVE": 10/12, "ORACLE": 10/12, "AI-CORR": 9/12, "LE-CORR": 9/12},
            {"CORR": 10/12, "NAIVE": 5/12, "ORACLE": 11/12, "AI-CORR": 9/12, "LE-CORR": 10/12},
        )
        maes = (
            {"CORR": .8985307652273619, "NAIVE": 3.1730864981559748, "ORACLE": .2542624655964872},
            {"CORR": .7608490819565342, "NAIVE": 3.034712517236455, "ORACLE": .12802468661625616},
        )
        with threadpool_limits(limits=1):
            truth, _, xi = population_reference(32)
        self.assertAlmostEqual(truth, .3624233067286206, places=12)
        for repetition in range(2):
            result = worker((32, repetition, truth, xi))
            actual = {row["method"]: row["accuracy"] for row in result["accuracy"]}
            self.assertEqual(actual, accuracies[repetition])
            for row in result["distance"]:
                np.testing.assert_allclose(row["mae"], maes[repetition][row["method"]],
                                           rtol=1e-7, atol=1e-9)
                self.assertEqual(row["pairs"], 24)

    def test_article_sample_counts(self):
        from simulation import DIMENSIONS, train_size, test_size
        self.assertEqual([train_size(p) for p in DIMENSIONS], [4, 5, 6, 7, 8, 9, 9, 10])
        self.assertEqual([test_size(p) for p in DIMENSIONS], [3, 4, 4, 5, 6, 6, 6, 7])
        with threadpool_limits(limits=1):
            _, labels, _, train, test = generate(32, np.random.default_rng(0))
        np.testing.assert_array_equal(np.bincount(labels[train]), [4]*4)
        np.testing.assert_array_equal(np.bincount(labels[test]), [3]*4)
        self.assertFalse(set(train) & set(test))

    def test_whitening_uses_training_objects_only(self):
        with threadpool_limits(limits=1):
            seed = np.random.SeedSequence([SEED, 32, 0]).spawn(2)[0]
            scms, _, sizes, train, test = generate(32, np.random.default_rng(seed))
            white, n_pool = whiten(scms, sizes, train)
            changed = scms.copy()
            changed[test] *= 10.
            altered, changed_pool = whiten(changed, sizes, train)
        np.testing.assert_array_equal(white[train], altered[train])
        self.assertEqual(n_pool, changed_pool)
        self.assertEqual(n_pool, 5120)


class VerificationTests(unittest.TestCase):
    def test_missing_and_duplicate_draws_are_rejected(self):
        reference = pd.DataFrame({"draw": [0, 1], "score": [.25, .75]})
        with self.assertRaises(ValueError):
            compare(reference.iloc[:1], reference, ["draw"], ["score"])
        with self.assertRaises(ValueError):
            compare(pd.concat([reference, reference.iloc[:1]]), reference, ["draw"], ["score"])
        result = compare(reference.iloc[:1], reference, ["draw"], ["score"], partial=True)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_rows"], 1)

    def test_equal_means_do_not_hide_different_draws(self):
        reference = pd.DataFrame({"draw": [0, 1], "score": [.25, .75]})
        altered = pd.DataFrame({"draw": [0, 1], "score": [.75, .25]})
        with self.assertRaises(AssertionError):
            compare(altered, reference, ["draw"], ["score"])

    def test_copied_histogram_cannot_hide_a_different_or_nonfinite_spectrum(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'figure1_histogram.csv').write_bytes(reference_file('figure1_histogram.csv').getvalue())
            values = np.r_[np.ones(254), 4.95, 7.74]
            np.savetxt(folder / 'figure1_eigenvalues.csv', values, delimiter=',', header='eigenvalue', comments='')
            with self.assertRaises(AssertionError):
                figure1(folder)
            values[0] = np.nan
            np.savetxt(folder / 'figure1_eigenvalues.csv', values, delimiter=',', header='eigenvalue', comments='')
            with self.assertRaises(ValueError):
                figure1(folder)


if __name__ == '__main__':
    unittest.main()
