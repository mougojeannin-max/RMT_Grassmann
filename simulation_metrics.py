"""The two corrected covariance distances used in the synthetic experiment."""

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import spence


def _affine_correction(lam, n_u, n_v):
    """Couillet et al. (2019), Corollary 4, for f(t)=log(t)^2."""
    p = len(lam)
    if n_u <= p or n_v <= p or np.any(lam <= 0):
        raise ValueError("AI-CORR requires positive spectra and sample sizes > p")
    c_u, c_v = p / n_u, p / n_v
    root = np.sqrt(lam)
    eta = np.linalg.eigvalsh(np.diag(lam) + np.outer(root, root) / (n_u - p))
    zeta = np.linalg.eigvalsh(np.diag(lam) - np.outer(root, root) / n_v)
    li, lj = lam[:, None], lam[None, :]
    delta = li - lj
    x = delta / lj
    lj_full = np.broadcast_to(lj, (p, p))
    small = np.abs(x) < 1e-4
    m, nn = np.empty((p, p)), np.empty((p, p))
    xs, ljs = x[small], lj_full[small]
    # Taylor limits remove cancellation near repeated eigenvalues.
    nn[small] = (1-xs/2+xs**2/3-xs**3/4+xs**4/5) / ljs
    m[small] = (.5-xs/3+xs**2/4-xs**3/5+xs**4/6) / ljs**2
    large = ~small
    lr = np.log1p(x[large])
    nn[large] = lr / delta[large]
    m[large] = (x[large] - lr) / delta[large]**2
    log_lam = np.log(lam) + np.log1p(-c_u)
    r = log_lam / lam
    eta_zeta, eta_lam = eta-zeta, eta-lam
    value = (
        np.dot(log_lam, log_lam) / p
        + 2*(c_u+c_v-c_u*c_v)/(c_u*c_v)
        * (eta_zeta @ m @ eta_lam + eta_lam @ r)
        - 2/p * np.sum(eta_zeta @ nn)
        - (1/c_v-1) * (np.log1p(-c_u)+np.log1p(-c_v))**2
        - 2*(1/c_v-1) * (eta_zeta @ r)
    )
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite AI-CORR estimate")
    return max(float(value), 0.)


def ai_distances(covariances, sizes, query, train):
    """Symmetrize the two oriented, zero-truncated AI corrections."""
    result = np.empty((len(query), len(train)), dtype=np.float32)
    for row, i in enumerate(query):
        a = np.asarray(covariances[i], dtype=float)
        factor = cholesky((a+a.T)*.5, lower=True, check_finite=True)
        for col, j in enumerate(train):
            b = np.asarray(covariances[j], dtype=float)
            b = (b+b.T)*.5
            temp = solve_triangular(factor, b, lower=True, check_finite=False)
            transformed = solve_triangular(factor, temp.T, lower=True,
                                           check_finite=False).T
            lam = np.linalg.eigvalsh((transformed+transformed.T)*.5)
            forward = _affine_correction(lam, int(sizes[i]), int(sizes[j]))
            reverse = _affine_correction(np.sort(1./lam), int(sizes[j]), int(sizes[i]))
            result[row, col] = .5*(forward+reverse)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite AI-CORR distance matrix")
    return result


def _phi2(x):
    """Real continuation of the dilogarithm in the LE estimator."""
    result = np.empty_like(x)
    below = x < 1.
    result[below] = spence(1.-x[below])
    result[~below] = (np.pi**2/3.-.5*np.log(x[~below])**2
                      - spence(1.-1./x[~below]))
    return result


def _le_representation(covariance, n):
    """Pereira et al. (2024), Section III-D, equations (23)-(24)."""
    values, vectors = np.linalg.eigh((covariance+covariance.T)*.5)
    p = len(values)
    if n <= p or np.any(values <= 0):
        raise ValueError("LE-CORR requires positive spectra and sample sizes > p")
    mu = np.linalg.eigvalsh(np.diag(values)-np.outer(np.sqrt(values), np.sqrt(values))/n)
    if np.any(mu <= 0):
        raise FloatingPointError("Nonpositive LE secular root")
    log_values, log_mu = np.log(values), np.log(mu)
    weights = np.empty(p)
    for k in range(p):
        mask = np.arange(p) != k
        other = values[mask]
        denominator = other-values[k]
        if np.any(denominator == 0):
            raise ValueError("LE-CORR requires simple empirical eigenvalues")
        weights[k] = (
            1.
            + (1.+np.sum(values[k]/denominator)
               - np.sum(mu[k]/(values-mu[k]))) * log_values[k]
            + np.sum(log_values[mask]*other/denominator)
            - np.sum(mu/(mu-values[k])*log_mu)
        )

    ratio = n/p
    aux1 = (ratio-1.)*np.sum(log_mu**2-log_values**2)
    aux2 = np.mean(log_values**2)+2.*np.mean(log_values)+2.
    log_one_minus_c = np.log1p(-p/n)
    aux3 = -(ratio-1.)*log_one_minus_c**2 + 2.*(ratio-1.)*log_one_minus_c
    lam_col = np.broadcast_to(values[:, None], (p, p))
    lam_row = np.broadcast_to(values[None, :], (p, p))
    gaps = np.abs(lam_row-lam_col)
    off_diagonal = ~np.eye(p, dtype=bool)
    term = np.zeros((p, p))
    term[off_diagonal] = (
        np.log(lam_row[off_diagonal]/lam_col[off_diagonal])
        * np.log(lam_col[off_diagonal]/gaps[off_diagonal])
    )
    aux4_1 = float(np.sum(term))
    mu_col = np.broadcast_to(mu[:, None], (p, p))
    aux4_2 = -float(np.sum(np.log(mu_col/lam_row)
                           * np.log(lam_row/np.abs(mu_col-lam_row))))
    aux5 = float(np.sum(_phi2(mu_col/lam_row)-_phi2(lam_col/lam_row)))
    alpha = aux1+aux2+aux3+2.*(aux4_1+aux4_2+aux5)/p
    representation = (vectors*weights)@vectors.T
    if not np.isfinite(alpha) or not np.isfinite(representation).all():
        raise FloatingPointError("Nonfinite LE-CORR representation")
    return alpha, (representation+representation.T)*.5


def le_distances(covariances, sizes, query, train):
    """Zero-truncated, normalized squared LE correction."""
    used = np.unique(np.concatenate((query, train)))
    representations = {int(i): _le_representation(covariances[i], int(sizes[i]))
                       for i in used}
    p = covariances.shape[1]
    result = np.empty((len(query), len(train)), dtype=np.float32)
    for row, i in enumerate(query):
        a_i, r_i = representations[int(i)]
        for col, j in enumerate(train):
            a_j, r_j = representations[int(j)]
            result[row, col] = max(float(a_i+a_j-2.*np.sum(r_i*r_j)/p), 0.)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite LE-CORR distance matrix")
    return result
