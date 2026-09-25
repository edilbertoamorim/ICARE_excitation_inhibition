"""Latent trajectory classes: a mixture of polynomial trajectories, fitted by EM.

The longitudinal counterpart of the latent class analysis in Calfee et al. (Lancet Respir Med
2014): a finite mixture in which every patient belongs to one of K latent classes, and each
class has its own mean trajectory over time per variable (Nagin's group-based / multi-
trajectory model). Variables are independent given the class, and a missing time point simply
drops out of the likelihood -- recordings start anywhere from hour 12 to hour 46, and nothing
here is ever fitted to interpolated values.

`re_terms` sets how much of a patient's own trajectory is allowed to differ from its class
mean -- the number of random-effect terms per patient and variable:

    y_itv = curve_kv(t) + z(t)' b_iv + e_itv,   b_iv ~ N(0, D_kv),  e_itv ~ N(0, sigma2_kv)

  * 0 -- no random effects: GBTM / LCGA in the strict sense. Any persistent patient-level
    offset has nowhere to go but into a new class, so on real data it over-extracts: BIC falls
    with every added class, posteriors saturate near 1, and the solution does not survive
    resampling. That is exactly what it did on the 52 CCMRC patients.
  * 1 -- random intercept. A patient who simply runs high is a shifted member of a class
    rather than a class of its own.
  * 2 -- random intercept + slope.
  * 3 -- random intercept + slope + quadratic. This is the specification Zack's GBTM uses
    through R `lcmm::hlme` (`random = ~ NormHour + I(NormHour^2)`), so `re_terms=3` with
    `degree=2` is the like-for-like setting for comparing against his classes.

The b_iv are integrated out analytically. Everything the EM needs is a per-patient sufficient
statistic, and the only matrices ever inverted are q x q with q = `re_terms` (<= 3), so the
cost is set by the number of patients rather than by the number of time points.

No I/O -- the calling step builds the (patient, time, variable) array and does the z-scoring.

Conventions:
  * Y is (n_patients, n_times, n_vars) with NaN where unobserved; t is (n_times,) and should
    be scaled to roughly [-1, 1] (or [0, 1], as Zack's NormHour is) so the basis is well
    conditioned.
  * BIC uses n = number of PATIENTS, not observations -- the patient is the sampling unit.
  * Entropy is the normalised (Mplus / Calfee) form: 1 = perfectly separated classes.

Self-check:  python3 -W ignore src/lcga.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VAR_FLOOR = 1e-3          # residual-variance floor (z units), stops a class collapsing on a point
D_RIDGE = 1e-6            # keeps the random-effect covariance invertible
RIDGE = 1e-8              # keeps the per-class normal equations solvable for a near-empty class
# Random-effect EM climbs slowly and the tail of that climb is NOT negligible here: on the
# 592-patient hourly fit, stopping at 2000 iterations cost 600 log-likelihood units at K=3 and
# only 7 at K=2, which inverted the ordering between them and would have broken every BIC
# comparison. The cap is therefore generous and the tolerance tight; `fit` warns if a refined
# run ever reaches the cap.
LL_TOL = 1e-9
MAX_ITER = 20_000
LOG2PI = float(np.log(2 * np.pi))
# Starts used by the self-check only. It is a correctness gate, not an analysis: enough starts
# to land in the right basin on 60 planted patients, not the 200 a real fit uses.
SELF_CHECK_STARTS = 12


@dataclass
class Fit:
    k: int
    degree: int
    re_terms: int
    loglik: float
    n_params: int
    bic: float
    aic: float
    entropy: float
    weights: np.ndarray       # (k,) class proportions
    beta: np.ndarray          # (k, n_vars, degree + 1) polynomial coefficients
    sigma2: np.ndarray        # (k, n_vars) residual variances
    D: np.ndarray             # (k, n_vars, q, q) random-effect covariance, q = re_terms
    post: np.ndarray          # (n, k) posterior class probabilities
    n_replicated: int         # refined starts that reached the best log-likelihood
    n_starts: int             # random starts screened
    n_refined: int            # of those, how many were run to convergence

    @property
    def modal(self) -> np.ndarray:
        return self.post.argmax(axis=1)

    @property
    def class_n(self) -> np.ndarray:
        return np.bincount(self.modal, minlength=self.k)

    @property
    def appa(self) -> np.ndarray:
        """Average posterior probability of the assigned class, per class (NaN if empty)."""
        m = self.modal
        return np.array([self.post[m == j, j].mean() if (m == j).any() else np.nan
                         for j in range(self.k)])

    def curve(self, t: np.ndarray) -> np.ndarray:
        """Class mean trajectories at times t: (k, len(t), n_vars)."""
        return np.einsum("tp,kvp->ktv", basis(t, self.degree), self.beta)


def basis(t: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(np.asarray(t, float), degree + 1, increasing=True)


def _prep(Y: np.ndarray, t: np.ndarray, degree: int, q: int):
    """Per-patient sufficient statistics: everything the EM needs that does not involve a class.

    Summing over time here is what keeps the fit cheap on hourly data -- no (class, patient,
    time) tensor is ever formed, so 592 patients x 61 hours costs the same as 592 x 10.
    """
    B = basis(t, degree)                                     # (T, p)
    obs = ~np.isnan(Y)                                       # (n, T, V)
    Yf = np.where(obs, Y, 0.0)
    of = obs.astype(float)
    st = {
        "B": B, "obs": obs, "Yf": Yf,
        "M": np.einsum("ntv,tp,tr->nvpr", of, B, B),         # sum_t B B'
        "r": np.einsum("ntv,tp->nvp", Yf, B),                # sum_t B y
        "yty": (Yf ** 2).sum(axis=1),                        # (n, V)
        "m": obs.sum(axis=1).astype(float),                  # (n, V) observed count
        "q": q,
    }
    if q:
        Z = basis(t, q - 1)                                  # (T, q)
        st["Z"] = Z
        st["ZtZ"] = np.einsum("ntv,ta,tb->nvab", of, Z, Z)
        st["ZtB"] = np.einsum("ntv,ta,tp->nvap", of, Z, B)
        st["Zty"] = np.einsum("ntv,ta->nva", Yf, Z)
    return st


def _sym_inv(M):
    """Inverse and log-determinant of a batch of symmetric q x q matrices, q <= 3.

    np.linalg.inv on a batch of 3x3 blocks spends ~70% of the EM's time in LAPACK call
    overhead -- there are k * n of them per iteration and each is tiny. The closed forms below
    are the same arithmetic as a cofactor expansion, done as elementwise numpy ops over the
    whole batch at once.
    """
    q = M.shape[-1]
    if q == 1:
        d = M[..., 0, 0]
        return (1.0 / d)[..., None, None], np.log(d)
    if q == 2:
        a, b, c, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
        det = a * d - b * c
        inv = np.stack([np.stack([d, -b], -1), np.stack([-c, a], -1)], -2) / det[..., None, None]
        return inv, np.log(det)
    if q == 3:
        a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
        d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
        g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
        A, B, C = e * i - f * h, -(d * i - f * g), d * h - e * g
        det = a * A + b * B + c * C
        inv = np.stack([np.stack([A, -(b * i - c * h), b * f - c * e], -1),
                        np.stack([B, a * i - c * g, -(a * f - c * d)], -1),
                        np.stack([C, -(a * h - b * g), a * e - b * d], -1)], -2)
        return inv / det[..., None, None], np.log(det)
    return np.linalg.inv(M), np.linalg.slogdet(M)[1]


def _err(st, beta):
    """Residual sums of squares per (class, patient, variable), from sufficient statistics."""
    return (st["yty"][None]
            - 2 * np.einsum("nvp,kvp->knv", st["r"], beta)
            + np.einsum("nvpr,kvp,kvr->knv", st["M"], beta, beta))


def _re_terms(st, beta, sigma2, D):
    """Random-effect quantities shared by the E-step and the likelihood.

    A = inv(D) + Z'Z/sigma2, so log|C| = m log sigma2 + log|D| + log|A| and
    e' inv(C) e = (e'e - (Z'e)' inv(A) (Z'e)/sigma2) / sigma2  (Woodbury + determinant lemma).
    A patient with no observations of a variable has Z'Z = 0, hence A = inv(D): its log|D| and
    log|A| cancel and its quadratic form is zero, so it drops out of the likelihood by itself.
    """
    Zte = st["Zty"][None] - np.einsum("nvap,kvp->knva", st["ZtB"], beta)
    Dinv, logdet_D = _sym_inv(D)                                     # (k, V, q, q)
    A = Dinv[:, None] + st["ZtZ"][None] / sigma2[:, None, :, None, None]
    Ainv, logdet_A = _sym_inv(A)
    return Zte, Ainv, logdet_D[:, None] + logdet_A


def _component_loglik(st, beta, sigma2, D) -> np.ndarray:
    """log p(y_i | class k), (n, k)."""
    ee = _err(st, beta)
    m, s2 = st["m"][None], sigma2[:, None, :]
    if st["q"]:
        Zte, Ainv, logdet = _re_terms(st, beta, sigma2, D)
        quad = (ee - np.einsum("knva,knvab,knvb->knv", Zte, Ainv, Zte) / s2) / s2
    else:
        logdet, quad = 0.0, ee / s2
    ll = -0.5 * (m * LOG2PI + m * np.log(s2) + logdet + quad)
    return ll.sum(axis=2).T


def _logsumexp(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True)))[:, 0]


def _m_step(post, st, beta, sigma2, D):
    """EM update. The random-effect moments come from the current parameters (ECM)."""
    k, q = post.shape[1], st["q"]
    n, V, p = st["M"].shape[0], st["M"].shape[1], st["M"].shape[2]

    if q and beta is not None:
        Zte, Ainv, _ = _re_terms(st, beta, sigma2, D)
        Eb = np.einsum("knvab,knvb->knva", Ainv, Zte) / sigma2[:, None, :, None]
        Vb = Ainv                                                # (k, n, V, q, q)
    else:
        Eb = np.zeros((k, n, V, max(q, 1)))
        Vb = np.zeros((k, n, V, max(q, 1), max(q, 1)))

    # beta: posterior-weighted least squares on y - Z E[b]
    beta = np.empty((k, V, p))
    for j in range(k):
        A = np.einsum("n,nvpr->vpr", post[:, j], st["M"]) + RIDGE * np.eye(p)
        rhs = np.einsum("n,nvp->vp", post[:, j], st["r"])
        if q:
            rhs = rhs - np.einsum("n,nvap,nva->vp", post[:, j], st["ZtB"], Eb[j])
        beta[j] = np.linalg.solve(A, rhs[..., None])[..., 0]

    ee = _err(st, beta)
    if q:
        Zte = st["Zty"][None] - np.einsum("nvap,kvp->knva", st["ZtB"], beta)
        bb = Eb[..., :, None] * Eb[..., None, :] + Vb            # E[b b']
        ee = (ee - 2 * np.einsum("knva,knva->knv", Eb, Zte)
              + np.einsum("nvab,knvab->knv", st["ZtZ"], bb))
    sigma2 = np.maximum(np.einsum("nk,knv->kv", post, ee)
                        / np.maximum(np.einsum("nk,nv->kv", post, st["m"]), 1e-12), VAR_FLOOR)
    if q and beta is not None:
        # only patients who have the variable at all inform its random-effect covariance
        has = (st["m"] > 0).astype(float)
        num = np.einsum("nk,nv,knvab->kvab", post, has, bb)
        den = np.maximum(np.einsum("nk,nv->kv", post, has), 1e-12)
        D = num / den[..., None, None] + D_RIDGE * np.eye(q)
    weights = np.clip(post.mean(axis=0), 1e-12, None)
    return weights / weights.sum(), beta, sigma2, D


def _pack(params):
    """Parameters -> a flat vector that any real value maps back from.

    log for the positive scalars and a Cholesky factor (log-diagonal) for D, so SQUAREM's
    extrapolation can never produce a negative variance or a non-PSD covariance.
    """
    w, beta, sigma2, D = params
    out = [np.log(w), beta.ravel(), np.log(sigma2).ravel()]
    if D.shape[-1] and D is not None and _has_re(D):
        L = np.linalg.cholesky(D)
        idx = np.tril_indices(D.shape[-1])
        diag = np.arange(D.shape[-1])
        Lv = L[..., idx[0], idx[1]].copy()
        is_diag = idx[0] == idx[1]
        Lv[..., is_diag] = np.log(L[..., diag, diag])
        out.append(Lv.ravel())
    return np.concatenate(out)


def _unpack(vec, shapes):
    """Inverse of `_pack`."""
    (k, V, p, q) = shapes
    i = 0
    w = np.exp(vec[i:i + k]); i += k
    w = np.clip(w, 1e-12, None); w = w / w.sum()
    beta = vec[i:i + k * V * p].reshape(k, V, p); i += k * V * p
    sigma2 = np.maximum(np.exp(vec[i:i + k * V]).reshape(k, V), VAR_FLOOR); i += k * V
    if q:
        nl = q * (q + 1) // 2
        Lv = vec[i:i + k * V * nl].reshape(k, V, nl)
        idx = np.tril_indices(q)
        is_diag = idx[0] == idx[1]
        L = np.zeros((k, V, q, q))
        vals = Lv.copy()
        vals[..., is_diag] = np.exp(np.clip(Lv[..., is_diag], -30, 30))
        L[..., idx[0], idx[1]] = vals
        D = L @ np.swapaxes(L, -1, -2) + D_RIDGE * np.eye(q)
    else:
        D = np.zeros((k, V, 1, 1))
    return w, beta, sigma2, D


def _has_re(D) -> bool:
    return D is not None and D.shape[-1] > 0 and not np.all(D == 0)


def _step(params, st):
    """One EM iteration as a fixed-point map: params -> params, plus the log-likelihood AT
    `params` and the posteriors it implies."""
    w, beta, sigma2, D = params
    joint = _component_loglik(st, beta, sigma2, D) + np.log(w)
    ll_i = _logsumexp(joint)
    post = np.exp(joint - ll_i[:, None])
    return _m_step(post, st, beta, sigma2, D), float(ll_i.sum()), post


def _init(post, st):
    """First M-step from a starting posterior (no random-effect moments to draw on yet)."""
    k, V, q = post.shape[1], st["M"].shape[1], st["q"]
    D = np.tile(0.1 * np.eye(max(q, 1)), (k, V, 1, 1))
    return _m_step(post, st, None, None, D)


def _em(post, st, max_iter=MAX_ITER, tol=LL_TOL, accelerate=True):
    """EM to convergence, accelerated by SQUAREM (Varadhan & Roland 2008).

    Plain EM needs ~3000 iterations on the 592-patient hourly fit, and truncating it inverted
    the BIC ordering between K=2 and K=3, so speed here is a correctness issue rather than a
    convenience. SQUAREM takes the two EM iterates from a point, extrapolates along their
    difference, and keeps the result only when the likelihood does not get worse -- so it can
    only ever reduce the iteration count, never change the fixed point it converges to.
    """
    params = _init(post, st)
    k, V, p = params[1].shape
    shapes = (k, V, p, st["q"])
    stepmax, ll, prev = 1.0, -np.inf, -np.inf
    hit_cap = True
    for _ in range(max_iter):
        p1, ll, post = _step(params, st)
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            hit_cap = False
            break
        prev = ll
        if not accelerate:
            params = p1
            continue
        p2, _, _ = _step(p1, st)
        x0, x1, x2 = _pack(params), _pack(p1), _pack(p2)
        r, v = x1 - x0, (x2 - x1) - (x1 - x0)
        nv = np.linalg.norm(v)
        if not np.isfinite(nv) or nv < 1e-12:
            params = p2
            continue
        alpha = float(np.clip(-np.linalg.norm(r) / nv, -stepmax, -1.0))
        cand = _unpack(x0 - 2 * alpha * r + alpha ** 2 * v, shapes)
        cand_next, ll_cand, _ = _step(cand, st)
        _, ll2, _ = _step(p2, st)
        if np.isfinite(ll_cand) and ll_cand >= ll2:
            params = cand_next
            if alpha <= -stepmax:
                stepmax = min(stepmax * 4.0, 1e4)
        else:
            params, stepmax = p2, max(1.0, stepmax / 4.0)
    w, beta, sigma2, D = params
    return ll, w, beta, sigma2, D, post, hit_cap


def _entropy(post: np.ndarray) -> float:
    n, k = post.shape
    if k == 1:
        return np.nan
    p = np.clip(post, 1e-300, 1.0)
    return float(1.0 - (-(p * np.log(p)).sum()) / (n * np.log(k)))


def fit(Y: np.ndarray, t: np.ndarray, k: int, degree: int = 2, n_starts: int = 200,
        seed: int = 0, re_terms: int = 0) -> Fit:
    """Best of `n_starts` random-posterior EM runs for a K-class trajectory mixture.

    EVERY start is run to convergence. Screening starts by their likelihood after a few dozen
    iterations -- lcmm's `gridsearch` strategy, and the obvious way to make this cheaper --
    was tried and is actively misleading here: on planted data the best-screened starts all
    led to the same inferior optimum (replicated 20/20, log-likelihood -441) while the good
    one (-409, and the planted class sizes) was reached by 2 starts in 40 that screening threw
    away. Early likelihood does not rank basins in this model, so the accelerated EM in `_em`
    is what buys the starts instead.
    """
    rng = np.random.default_rng(seed)
    q = int(re_terms)
    st = _prep(Y, t, degree, q)
    n, V = Y.shape[0], Y.shape[2]

    if k == 1:
        runs = [_em(np.ones((n, 1)), st)]
    else:
        runs = [_em(rng.dirichlet(np.ones(k), size=n), st) for _ in range(n_starts)]
    lls = [r[0] for r in runs]
    ll, w, beta, sigma2, D, post, hit_cap = max(runs, key=lambda r: r[0])
    if hit_cap:
        import warnings
        warnings.warn(f"EM hit the {MAX_ITER}-iteration cap at K={k} (re_terms={q}): the "
                      "log-likelihood may still be climbing, so BIC across K is not safe")
    n_re = q * (q + 1) // 2                          # free entries of a symmetric D
    n_params = (k - 1) + k * V * (degree + 1) + k * V * (1 + n_re)
    n_rep = int(np.sum(np.asarray(lls) >= ll - 1e-4 * max(1.0, abs(ll))))
    return Fit(k=k, degree=degree, re_terms=q, loglik=ll, n_params=n_params,
               bic=-2 * ll + n_params * np.log(n), aic=-2 * ll + 2 * n_params,
               entropy=_entropy(post), weights=w, beta=beta, sigma2=sigma2, D=D, post=post,
               n_replicated=n_rep, n_starts=len(lls), n_refined=len(lls))


def posterior(model: Fit, Y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Class posteriors for (possibly new) patients under fixed model parameters."""
    st = _prep(Y, t, model.degree, model.re_terms)
    joint = (_component_loglik(st, model.beta, model.sigma2, model.D)
             + np.log(model.weights))
    return np.exp(joint - _logsumexp(joint)[:, None])


def simulate(model: Fit, mask: np.ndarray, t: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Draw a dataset from `model` with the observation pattern `mask` (n, T, V)."""
    n, V = mask.shape[0], mask.shape[2]
    z = rng.choice(model.k, size=n, p=model.weights)
    y = model.curve(t)[z]                                    # (n, T, V)
    if model.re_terms:
        Z = basis(t, model.re_terms - 1)
        for v in range(V):
            L = np.linalg.cholesky(model.D[z, v])            # (n, q, q)
            b = np.einsum("nab,nb->na", L, rng.normal(size=(n, model.re_terms)))
            y[:, :, v] += b @ Z.T
    y = y + rng.normal(size=y.shape) * np.sqrt(model.sigma2[z])[:, None, :]
    return np.where(mask, y, np.nan), z


def blrt(Y: np.ndarray, t: np.ndarray, fit_k: Fit, fit_km1: Fit, n_boot: int = 100,
         n_starts: int = 20, seed: int = 0) -> tuple[float, float]:
    """Parametric bootstrap LRT of K vs K-1 classes: (observed LR, p).

    Datasets are simulated under the K-1 model with the real missingness pattern (every
    patient keeps their own observed times), and both models are refitted to each. The
    resampling counterpart of the VLMR test Calfee et al. report.
    """
    rng = np.random.default_rng(seed)
    lr_obs = 2 * (fit_k.loglik - fit_km1.loglik)
    mask = ~np.isnan(Y)
    exceed = 0
    for _ in range(n_boot):
        ys, _ = simulate(fit_km1, mask, t, rng)
        s = int(rng.integers(1 << 31))
        l0 = fit(ys, t, fit_km1.k, fit_k.degree, n_starts, s, fit_k.re_terms).loglik
        l1 = fit(ys, t, fit_k.k, fit_k.degree, n_starts, s + 1, fit_k.re_terms).loglik
        exceed += (2 * (l1 - l0)) >= lr_obs
    return float(lr_obs), float((1 + exceed) / (1 + n_boot))


# ---- self-check -----------------------------------------------------------------------

def _planted(rng, offset_sd: float = 0.0, slope_sd: float = 0.0):
    """60 patients, three planted bivariate shapes, cohort-like ragged coverage."""
    t = np.linspace(-1, 1, 10)
    n, T = 60, len(t)
    true = np.repeat([0, 1, 2], n // 3)
    shapes = np.array([[[-1.0, 1.0, 0.0], [0.5, 0.0, 0.0]],     # low -> up, Y flat high
                       [[1.0, -1.0, 0.0], [-0.5, 0.0, 0.0]],    # high -> down, Y flat low
                       [[1.0, 0.0, -0.8], [0.0, -0.8, 0.0]]])   # high hump, Y falling
    Y = np.einsum("tp,nvp->ntv", basis(t, 2), shapes[true])
    Y += rng.normal(scale=offset_sd, size=(n, 1, 2))            # persistent patient offsets
    if slope_sd:
        # drawn only when asked for: a zero-scale draw still consumes the random stream, which
        # would silently change every planted dataset built before this argument existed.
        Y += rng.normal(scale=slope_sd, size=(n, 1, 2)) * t[None, :, None]   # personal slopes
    Y += rng.normal(scale=0.35, size=Y.shape)
    start = rng.integers(0, 6, size=n)
    stop = np.maximum(rng.integers(6, T + 1, size=n), start + 3)
    idx = np.arange(T)
    keep = (idx[None] >= start[:, None]) & (idx[None] < stop[:, None])
    keep &= rng.random((n, T)) > 0.15
    Y[~keep] = np.nan
    return Y, t, true, shapes


def _oracle(shapes, post, re_terms, D_diag=0.0, sigma=0.35) -> Fit:
    q = max(re_terms, 1)
    return Fit(k=3, degree=2, re_terms=re_terms, loglik=np.nan, n_params=0, bic=np.nan,
               aic=np.nan, entropy=np.nan, weights=np.full(3, 1 / 3), beta=shapes,
               sigma2=np.full((3, 2), sigma ** 2),
               D=np.tile(D_diag * np.eye(q), (3, 2, 1, 1)), post=post, n_replicated=0,
               n_starts=0, n_refined=0)


def test_loglik_matches_dense() -> None:
    """The Woodbury likelihood must equal the textbook dense one, for every re_terms.

    `_component_loglik` never forms a patient's covariance matrix: it works from q x q pieces
    (Woodbury + the determinant lemma) so that hourly data costs the same as binned data. That
    identity is the one piece of maths in this module that is easy to get subtly wrong and
    impossible to notice downstream, so it is checked directly against
    scipy.stats.multivariate_normal on C = sigma2 I + Z D Z'.
    """
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(0)
    Y, t, _, _ = _planted(rng, offset_sd=0.3, slope_sd=0.2)
    k, V, degree = 3, Y.shape[2], 2
    for q in (0, 1, 2, 3):
        st = _prep(Y, t, degree, q)
        beta = rng.normal(size=(k, V, degree + 1))
        sigma2 = rng.uniform(0.2, 0.6, size=(k, V))
        L = rng.normal(size=(k, V, max(q, 1), max(q, 1))) * 0.3
        D = L @ np.swapaxes(L, -1, -2) + 0.2 * np.eye(max(q, 1))
        got = _component_loglik(st, beta, sigma2, D)

        B, Z = basis(t, degree), basis(t, max(q, 1) - 1)
        want = np.zeros_like(got)
        for i in range(Y.shape[0]):
            for j in range(k):
                total = 0.0
                for v in range(V):
                    obs = ~np.isnan(Y[i, :, v])
                    if not obs.any():
                        continue
                    mean = B[obs] @ beta[j, v]
                    C = sigma2[j, v] * np.eye(int(obs.sum()))
                    if q:
                        C = C + Z[obs] @ D[j, v] @ Z[obs].T
                    total += multivariate_normal(mean, C).logpdf(Y[i, obs, v])
                want[i, j] = total
        assert np.allclose(got, want, atol=1e-8), (
            f"re_terms={q}: Woodbury likelihood differs from the dense one by "
            f"{np.abs(got - want).max():.2e}")
    print("log-likelihood matches the dense multivariate normal for re_terms=0,1,2,3")


def test_recovers_planted_classes() -> None:
    """Planted classes must come back, and random effects must earn their keep.

    Recovery is judged against the ARI the TRUE parameters give, not against 1: a patient seen
    for three times can be genuinely ambiguous, so the oracle is the ceiling.
    """
    from sklearn.metrics import adjusted_rand_score as ari

    # 1. No patient-level variation: the fixed-effects mixture is the true model.
    Y, t, true, shapes = _planted(np.random.default_rng(1))
    fits = {k: fit(Y, t, k, n_starts=SELF_CHECK_STARTS, seed=k) for k in range(1, 6)}
    best_k = min(fits, key=lambda k: fits[k].bic)
    f3 = fits[3]
    a = ari(true, f3.modal)
    a_oracle = ari(true, posterior(_oracle(shapes, f3.post, 0), Y, t).argmax(axis=1))
    match = [np.bincount(f3.modal[true == j], minlength=3).argmax() for j in range(3)]
    err = np.abs(f3.beta[match] - shapes).max()
    assert best_k == 3, f"BIC picked K={best_k}, expected 3"
    assert a >= a_oracle - 0.05, f"ARI {a:.2f} vs {a_oracle:.2f} with true parameters"
    assert err < 0.3, f"planted curve coefficients recovered only to within {err:.2f}"
    assert np.allclose(posterior(f3, Y, t), f3.post, atol=1e-6), "posterior() != fit"
    print(f"re_terms=0, no patient variation: BIC picks K={best_k}, ARI={a:.3f} "
          f"(true parameters {a_oracle:.3f}), max coefficient error {err:.2f}, best LL "
          f"replicated by {f3.n_replicated}/{f3.n_refined} refined starts")

    # 2. The optimiser must not do WORSE than the random-intercept-only implementation this
    # module replaced. Frozen optima from that version, on a dataset it was run on; the fit is
    # free to beat them (screening + a tighter tolerance do), never to fall short.
    Yr, tr, _, _ = _planted(np.random.default_rng(7), offset_sd=0.5)
    for q, k, before in [(0, 2, -605.2895102712508), (0, 3, -545.339617075944),
                         (1, 2, -434.5629431506958), (1, 3, -374.1409601105117)]:
        got = fit(Yr, tr, k, 2, 25, 11, q).loglik
        assert got >= before - 1e-6 * abs(before), (
            f"re_terms={q}, K={k}: log-likelihood {got:.4f} is below the {before:.4f} the "
            "previous implementation reached")
    print("re_terms=0 and 1 still reach the pre-rewrite optima")

    # 3. Persistent offsets: a random intercept must keep BIC from over-extracting.
    sd = 0.4
    Y, t, true, shapes = _planted(np.random.default_rng(2), offset_sd=sd)
    lc = {k: fit(Y, t, k, n_starts=SELF_CHECK_STARTS, seed=k).bic for k in range(1, 6)}
    gm = {k: fit(Y, t, k, n_starts=SELF_CHECK_STARTS, seed=k, re_terms=1)
          for k in range(1, 6)}
    k_lc, k_gm = min(lc, key=lc.get), min(gm, key=lambda k: gm[k].bic)
    a = ari(true, gm[3].modal)
    a_oracle = ari(true, posterior(_oracle(shapes, gm[3].post, 1, sd ** 2), Y, t).argmax(axis=1))
    assert k_gm == 3, f"random intercept: BIC picked K={k_gm} with planted offsets, expected 3"
    assert a >= a_oracle - 0.1, f"ARI {a:.2f} vs {a_oracle:.2f} with true parameters"
    print(f"planted offsets (sd {sd}): re_terms=1 picks K={k_gm} (ARI {a:.3f}, true "
          f"parameters {a_oracle:.3f}), re_terms=0 picks K={k_lc}")

    # 4. Personal slopes as well -- Zack's specification (intercept + slope + quadratic).
    # Here BIC is NOT asserted to land on 3: once each patient carries their own slope, a
    # 2-class solution plus wide random effects explains the same data, and at n=60 that is a
    # real identifiability limit rather than a bug. What is asserted is that the q=3 path
    # estimates a full 3x3 covariance and separates the planted classes at least as well as
    # the fixed-effects model does on the same data.
    Y, t, true, shapes = _planted(np.random.default_rng(3), offset_sd=0.35, slope_sd=0.35)
    fx = {k: fit(Y, t, k, n_starts=SELF_CHECK_STARTS, seed=k) for k in range(1, 5)}
    re3 = {k: fit(Y, t, k, n_starts=SELF_CHECK_STARTS, seed=k, re_terms=3)
           for k in range(1, 5)}
    k_fx = min(fx, key=lambda k: fx[k].bic)
    k_re3 = min(re3, key=lambda k: re3[k].bic)
    a_fx, a_re3 = ari(true, fx[3].modal), ari(true, re3[3].modal)
    assert re3[3].D.shape[-1] == 3, "re_terms=3 must estimate a 3x3 random-effect covariance"
    assert np.all(np.linalg.eigvalsh(re3[3].D) > 0), "random-effect covariance is not PSD"
    assert a_re3 >= a_fx, (f"with planted slopes, re_terms=3 recovers the classes worse "
                           f"(ARI {a_re3:.2f}) than re_terms=0 (ARI {a_fx:.2f})")
    print(f"planted offsets + slopes: at K=3 re_terms=3 ARI {a_re3:.3f} vs re_terms=0 "
          f"{a_fx:.3f}; BIC picks K={k_re3} with random effects, K={k_fx} without")


if __name__ == "__main__":
    test_loglik_matches_dense()
    test_recovers_planted_classes()
