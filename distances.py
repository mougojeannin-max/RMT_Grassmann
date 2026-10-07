"""The two completed Grassmann distances and the two PCA baselines."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist


_DISTANCE_THREADS = 1
_DISTANCE_EXECUTOR = None


def configure_distance_threads(n):
    """Configure a persistent pool once, before starting distance calculations."""
    global _DISTANCE_THREADS, _DISTANCE_EXECUTOR
    if not isinstance(n, int) or n < 1:
        raise ValueError('Distance thread count must be a positive integer')
    if n == _DISTANCE_THREADS:
        return
    if _DISTANCE_EXECUTOR is not None:
        _DISTANCE_EXECUTOR.shutdown(wait=True)
    _DISTANCE_THREADS = n
    _DISTANCE_EXECUTOR = ThreadPoolExecutor(max_workers=n) if n > 1 else None


def rank_threshold(p, n, n_global=None, alpha=1 / 3, beta=1 / 3):
    """Upper Marchenko--Pastur edge with the finite-sample margins."""
    if p <= 0 or n <= 0 or not 0 < alpha < 2 / 3 or not 0 < beta < .5:
        raise ValueError('Invalid dimensions or threshold exponents')
    margin = 0 if n_global is None else (p / float(n_global)) ** beta
    return (1 + np.sqrt(p / float(n))) ** 2 + p ** (-alpha) + margin


def alignment_coefficients(values, c, noise, p):
    """Estimate eigenvector alignment, with the article's ambient-dimension floor."""
    if c <= 0 or noise <= 0 or not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError('Positive aspect ratio, noise estimate and integer dimension required')
    centered = np.asarray(values) - (1 + c) * noise
    discriminant = centered ** 2 - 4 * c * noise ** 2
    spikes = .5 * (centered + np.sqrt(np.maximum(discriminant, 0)))
    if np.any(spikes <= 0) or not np.isfinite(spikes).all():
        raise ValueError('Invalid estimated spikes')
    xi = (1 - c * noise ** 2 / spikes ** 2) / (1 + c * noise / spikes)
    return np.maximum(xi, 1 / p)


def _singular_values(grams, corrected):
    # For large uncorrected subspaces, the smaller Gram matrix is faster.
    if not corrected and min(grams.shape[1:]) >= 32:
        small = (grams @ grams.swapaxes(-1, -2) if grams.shape[1] <= grams.shape[2]
                 else grams.swapaxes(-1, -2) @ grams)
        return np.sqrt(np.maximum(np.linalg.eigvalsh(small), 0))[..., ::-1]
    return np.linalg.svd(grams, compute_uv=False)


def completed_distances(bases, query, train, xis=None, complements=None):
    """Principal-angle distance plus pi^2/4 per unmatched dimension.

    With ``xis``, correct the overlap before computing its singular values.
    Orthogonal complements only accelerate the uncorrected calculation.
    """
    bases = tuple(np.asarray(b, dtype=float) for b in bases)
    if not bases or any(b.ndim != 2 or b.shape[0] != bases[0].shape[0] for b in bases):
        raise ValueError('Bases must share an ambient space')
    if xis is not None:
        xis = tuple(np.asarray(x, dtype=float) for x in xis)
        if len(xis) != len(bases) or any(x.shape != (b.shape[1],) or
                np.any(x <= 0) or not np.isfinite(x).all() for b, x in zip(bases, xis)):
            raise ValueError('Positive alignment coefficients must match each rank')
    p = bases[0].shape[0]
    ranks = np.asarray([b.shape[1] for b in bases])
    query, train = np.asarray(query, dtype=int), np.asarray(train, dtype=int)
    if complements is not None:
        complements = tuple(np.asarray(b, dtype=float) for b in complements)
        if len(complements) != len(bases) or any(b.shape != (p, p - r)
                for b, r in zip(complements, ranks)):
            raise ValueError('Invalid orthogonal-complement dimensions')
    groups = []
    for rank in np.unique(ranks[train]):
        positions = np.flatnonzero(ranks[train] == rank)
        ids = train[positions]
        use_complement = ((np.minimum(p - ranks[query], p - rank)
                           < np.minimum(ranks[query], rank))
                          if xis is None and complements is not None
                          else np.zeros(len(query), dtype=bool))
        # A query group may need bases, complements, or both; keep only used stacks.
        references = np.stack([bases[i] for i in ids]) if not use_complement.all() else None
        complement_refs = (np.stack([complements[i] for i in ids])
                           if use_complement.any() else None)
        groups.append((rank, positions, ids, references,
                       np.stack([xis[i] for i in ids]) if xis is not None else None,
                       complement_refs))
    matrix = np.empty((len(query), len(train)), dtype=np.float32)
    def compute_row(item):
        row, i = item
        for rank, positions, ids, references, xi_refs, complement_refs in groups:
            use_complement = (xis is None and complements is not None and
                              min(p - ranks[i], p - rank) < min(ranks[i], rank))
            u = complements[i] if use_complement else bases[i]
            v = complement_refs if use_complement else references
            angles = np.zeros(len(positions))
            if min(u.shape[1], v.shape[2]):
                gram = np.einsum('pa,npb->nab', u, v, optimize=True)
                if xis is not None:
                    gram /= np.sqrt(xis[i][None, :, None] * xi_refs[:, None, :])
                singular = np.clip(_singular_values(gram, xis is not None), 0, 1)
                angles = np.sum(np.arccos(singular) ** 2, axis=1)
            if use_complement and np.any(ids == i):
                gram = np.einsum('pa,pb->ab', bases[i], bases[i], optimize=True)[None]
                singular = np.clip(_singular_values(gram, False)[0], 0, 1)
                angles[ids == i] = np.sum(np.arccos(singular) ** 2)
            matrix[row, positions] = np.sqrt(angles + (np.pi ** 2 / 4) * abs(ranks[i] - rank))
    if _DISTANCE_EXECUTOR is None or len(query) < 2:
        for item in enumerate(query):
            compute_row(item)
    else:
        list(_DISTANCE_EXECUTOR.map(compute_row, enumerate(query)))
    if not np.isfinite(matrix).all():
        raise FloatingPointError('Nonfinite Grassmann distances')
    return matrix


def pca_distances(model, method):
    """Affine-invariant or log-Euclidean distance, divided by sqrt(PCA rank)."""
    cov, refs, query = model['projected'], model['train'], model['query']
    used = np.unique(np.r_[refs, query])
    rank = model['pca_rank']
    if method == 'AI_PCA':
        factors = {int(i): cholesky(cov[i], lower=True) for i in used}
        reference_factors = np.stack([factors[int(i)] for i in refs])
        rhs = reference_factors.transpose(1, 0, 2).reshape(rank, -1)
        matrix = np.empty((len(query), len(refs)))
        for row, i in enumerate(query):
            relative = solve_triangular(factors[int(i)], rhs, lower=True, check_finite=False)
            relative = relative.reshape(rank, len(refs), rank).transpose(1, 0, 2)
            singular = np.linalg.svd(relative, compute_uv=False)
            if np.any(singular <= 0):
                raise ValueError('Singular relative PCA factor')
            matrix[row] = np.sqrt(np.mean((2 * np.log(singular)) ** 2, axis=1))
        return matrix
    if method != 'LE_PCA':
        raise ValueError('Unknown PCA distance')
    ev, vec = np.linalg.eigh(cov[used])
    if np.any(ev <= 0) or not np.isfinite(ev).all():
        raise ValueError('Non-SPD PCA covariance; no ridge or pseudoinverse')
    logs = (vec * np.log(ev)[:, None, :]) @ vec.swapaxes(-1, -2)
    flat = logs.reshape(len(used), -1)
    return cdist(flat[np.searchsorted(used, query)], flat[np.searchsorted(used, refs)]) / np.sqrt(rank)
