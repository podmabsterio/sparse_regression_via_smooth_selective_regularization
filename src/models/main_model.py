import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import time


def pen_H(x, lambda_, M):
    return M * x**2 * np.exp(-lambda_ * x**2 * 0.5)


def pen_H_der2(x, lambda_, M):
    return (
        M * (2 - 5 * lambda_ * x**2 + lambda_**2 * x**4) * np.exp(-lambda_ * x**2 * 0.5)
    )


def precalculated_exp_part(x, lambda_):
    return np.exp(-lambda_ * x**2 * 0.5)


def pen_H_der1_precalculated(x, lambda_, M, exp_part):
    return M * (2 * x - lambda_ * x**3) * exp_part


def pen_H_der2_precalculated(x, lambda_, M, exp_part):
    return M * (2 - 5 * lambda_ * x**2 + lambda_**2 * x**4) * exp_part


def pen_H_der3_precalculated(x, lambda_, M, exp_part):
    return (
        M * (-12 * lambda_ * x + 9 * lambda_**2 * x**3 - lambda_**3 * x**5) * exp_part
    )


def loss_function_der1(x, A, B, Lambda, M, exp_part):
    return 2 * A * x + B + pen_H_der1_precalculated(x, Lambda, M, exp_part)


def loss_function_der2(x, A, Lambda, M, exp_part):
    return 2 * A + pen_H_der2_precalculated(x, Lambda, M, exp_part)


def batch_find_convex_bounds(
    x1,
    x2,
    Lambda,
    M,
    A,
    close_to_left: bool,
    bias=0.1,
    max_iters=30,
    eps=1e-9,
    delta=1e-8,
    mu=1e-8,
):
    x_l = np.full(A.shape[0], x1)
    x_r = np.full(A.shape[0], x2)

    converged = np.zeros(A.shape[0], dtype=np.bool_)

    if close_to_left:
        x0 = x1 + bias * (x2 - x1)
    else:
        x0 = x2 - bias * (x2 - x1)

    x = np.full(A.shape[0], x0)

    for _ in range(max_iters):
        not_converged = ~converged
        A_nc = A[not_converged]
        x_nc = x[not_converged]
        x_l_nc = x_l[not_converged]
        x_r_nc = x_r[not_converged]

        exp_part = precalculated_exp_part(x_nc, Lambda)
        f2 = loss_function_der2(x_nc, A_nc, Lambda, M, exp_part)
        f3 = pen_H_der3_precalculated(x_nc, Lambda, M, exp_part)

        xN = x_nc - f2 / f3

        safe_newton = (xN > x_l_nc + delta) & (xN < x_r_nc - delta) & (np.abs(f3) > mu)

        x_new = np.where(safe_newton, xN, 0.5 * (x_l_nc + x_r_nc))

        f2_new = pen_H_der2(x_new, Lambda, M)
        left_mask = f2_new > 0
        x_l[not_converged] = np.where(left_mask, x_new, x_l_nc)
        x_r[not_converged] = np.where(~left_mask, x_new, x_r_nc)

        converged_now = np.abs(x_new - x_nc) < eps
        x[not_converged] = x_new
        converged[not_converged] = converged_now

        if np.all(converged):
            return x

    return x


def batch_safeguarded_newton(
    Lambda,
    M,
    A,
    B,
    x0,
    left_bound,
    right_bound,
    max_iters=20,
    eps=1e-12,
    delta=1e-8,
    mu=1e-8,
):
    x = x0.copy()
    x_l = np.full_like(x, left_bound)
    x_r = np.full_like(x, right_bound)

    converged = np.zeros(x.shape[0], dtype=np.bool_)

    for _ in range(max_iters):
        not_converged = ~converged
        x_nc = x[not_converged]
        x_l_nc = x_l[not_converged]
        x_r_nc = x_r[not_converged]
        A_nc = A[not_converged]

        exp_part = precalculated_exp_part(x_nc, Lambda)
        fp = loss_function_der1(x_nc, A_nc, B[not_converged], Lambda, M, exp_part)
        fpp = loss_function_der2(x_nc, A_nc, Lambda, M, exp_part)

        xN = x_nc - fp / fpp

        safe = (xN > x_l_nc + delta) & (xN < x_r_nc - delta) & (np.abs(fpp) > mu)

        x_new = np.where(safe, xN, 0.5 * (x_l_nc + x_r_nc))

        fp_new = loss_function_der1(
            x_new,
            A_nc,
            B[not_converged],
            Lambda,
            M,
            precalculated_exp_part(x_new, Lambda),
        )
        left = fp_new < 0
        x_l[not_converged] = np.where(left, x_new, x_l_nc)
        x_r[not_converged] = np.where(~left, x_new, x_r_nc)

        converged_now = np.abs(x_new - x_nc) < eps
        x[not_converged] = x_new
        converged[not_converged] = converged_now

        if np.all(converged):
            return x

    return x


def get_approx_bounds(Lambda):
    return [
        (
            -(((5 + 17**0.5) / (2 * Lambda)) ** 0.5),
            -(((9 - 33**0.5) / (2 * Lambda)) ** 0.5),
        ),
        (
            -(((9 - 33**0.5) / (2 * Lambda)) ** 0.5),
            -(((5 - 17**0.5) / (2 * Lambda)) ** 0.5),
        ),
        (((5 - 17**0.5) / (2 * Lambda)) ** 0.5, ((9 - 33**0.5) / (2 * Lambda)) ** 0.5),
        (((9 - 33**0.5) / (2 * Lambda)) ** 0.5, ((5 + 17**0.5) / (2 * Lambda)) ** 0.5),
    ]


def find_all_convex_bounds(Lambda, M, A, maxiter=20, eps=1e-12, max_coef_value=10):
    approx_bounds = get_approx_bounds(Lambda)

    x1 = batch_find_convex_bounds(
        approx_bounds[0][0], approx_bounds[0][1], Lambda, M, A, True, eps=eps
    )
    x2 = batch_find_convex_bounds(
        approx_bounds[1][0], approx_bounds[1][1], Lambda, M, A, False, eps=eps
    )
    x3 = batch_find_convex_bounds(
        approx_bounds[2][0], approx_bounds[2][1], Lambda, M, A, True, eps=eps
    )
    x4 = batch_find_convex_bounds(
        approx_bounds[3][0], approx_bounds[3][1], Lambda, M, A, False, eps=eps
    )

    inf_bound = np.full(A.shape[0], max_coef_value)
    return [(-inf_bound, x1), (x2, x3), (x4, inf_bound)]


def safe_newton_multistart(A, B, Lambda, M, maxiter, tol, convex_bounds):
    p = A.shape[0]
    starts = get_starts(Lambda)
    f_min = np.full((len(starts), p), np.inf)
    x_min = np.full((len(starts), p), np.inf)

    for i, (start, bounds) in enumerate(zip(starts, convex_bounds)):
        x0 = np.full(p, start)
        x_min[i] = batch_safeguarded_newton(
            Lambda,
            M,
            A,
            B,
            x0,
            bounds[0],
            bounds[1],
            max_iters=maxiter,
            eps=tol,
            delta=1e-8,
            mu=1e-8,
        )
        f_min[i] = A * x_min[i] ** 2 + B * x_min[i] + pen_H(x_min[i], Lambda, M)

    optimal_start_idx = np.argmin(f_min, axis=0)
    optimal_x = x_min[optimal_start_idx, np.arange(p)]
    optimal_f = f_min[optimal_start_idx, np.arange(p)]
    return optimal_x, optimal_f


def coord_coeffs(X: np.ndarray, Y: np.ndarray, theta: np.ndarray):
    A = np.sum(X * X, axis=0)
    Xtheta = X @ theta
    Qtheta = X.T @ Xtheta
    XY = X.T @ Y
    B = 2.0 * (Qtheta - A * theta) - 2.0 * XY
    return A, B


def get_starts(lambda_):
    x2 = ((9 + 33**0.5) / (2 * lambda_)) ** 0.5
    return np.array([-x2, 0.0, x2])


def unsafe_batch_newton(A, B, Lambda, M, x, maxiter, tol, tol_grad):
    eps = 1e-8
    converged = np.zeros(x.shape[0], dtype=np.bool_)

    for iter in range(maxiter):
        not_converged = ~converged

        exp_part = precalculated_exp_part(x[not_converged], Lambda)
        x_nc = x[not_converged]
        g = (
            2 * A[not_converged] * x_nc
            + B[not_converged]
            + pen_H_der1_precalculated(x_nc, Lambda, M, exp_part)
        )
        h = 2 * A[not_converged] + pen_H_der2_precalculated(x_nc, Lambda, M, exp_part)

        h = np.where(np.abs(h) < eps, eps, h)

        x_new = x_nc - g / h

        converged_now = (np.abs(x_new - x_nc) < tol) | (np.abs(g) < tol_grad)

        x[not_converged] = x_new
        converged[not_converged] = converged_now
        if np.all(converged):
            return x

    return x


def unsafe_batch_newton_multistart(A, B, Lambda, M, maxiter, tol, tol_grad):
    p = A.shape[0]
    starts = get_starts(Lambda)
    f_min = np.full((len(starts), p), np.inf)
    x_min = np.full((len(starts), p), np.inf)
    for i, start in enumerate(starts):
        x0 = np.full(p, start)
        x_min[i] = unsafe_batch_newton(A, B, Lambda, M, x0, maxiter, tol, tol_grad)
        f_min[i] = A * x_min[i] ** 2 + B * x_min[i] + pen_H(x_min[i], Lambda, M)

    optimal_start_idx = np.argmin(f_min, axis=0)
    optimal_x = x_min[optimal_start_idx, np.arange(p)]
    optimal_f = f_min[optimal_start_idx, np.arange(p)]
    return optimal_x, optimal_f


def choose_coord_for_update(update, strategy="max", k=5):
    n = update.shape[0]

    if strategy == "max":
        return int(np.argmax(update))

    k = min(k, n)
    topk_idxs = np.argpartition(update, -k)[-k:]

    if strategy == "topk_linear":
        vals = update[topk_idxs]
        min_val = vals.min()
        if min_val < 0:
            vals = vals - min_val
        total = vals.sum()
        if total == 0:
            probs = np.ones_like(vals) / len(vals)
        else:
            probs = vals / total
        return int(np.random.choice(topk_idxs, p=probs))

    elif strategy == "topk_softmax":
        vals = update[topk_idxs]
        exps = np.exp(vals - np.max(vals))
        probs = exps / exps.sum()
        return int(np.random.choice(topk_idxs, p=probs))

    else:
        raise ValueError(f"Unknown strategy for choosing coordinate: {strategy!r}")


def fit_main_model(
    X,
    y,
    Lambda,
    M,
    tol=1e-8,
    maxiter=1000,
    single_tol=1e-8,
    single_maxiter=50,
    single_tol_grad=1e-8,
    save_history=False,
    use_safe=False,
    init=None,
    strategy="max",
    strategy_k=5,
):
    if init is None:
        coef = np.zeros(X.shape[1])
    else:
        coef = init

    if save_history:
        history = np.zeros(maxiter)
    else:
        history = None

    A, B = coord_coeffs(X, y, coef)
    XtX = X.T @ X
    bounds = None

    if use_safe:
        assert (
            np.max(A) < 1.34 * M
        ), "One of the 1d problems is fully convex. Cannot use safe newton."
        bounds = find_all_convex_bounds(Lambda, M, A)

    converged = False
    for iter in range(maxiter):
        f_initial = A * coef**2 + B * coef + pen_H(coef, Lambda, M)

        if use_safe:
            optimal_x, optimal_f = safe_newton_multistart(
                A, B, Lambda, M, single_maxiter, single_tol, bounds
            )
        else:
            optimal_x, optimal_f = unsafe_batch_newton_multistart(
                A, B, Lambda, M, single_maxiter, single_tol, single_tol_grad
            )

        update = f_initial - optimal_f
        update_idx = choose_coord_for_update(update, strategy=strategy, k=strategy_k)
        update_value = optimal_x[update_idx]
        best_update = update[update_idx]
        if best_update < tol:
            converged = True
            break

        delta = update_value - coef[update_idx]
        col = XtX[:, update_idx]
        B = B + 2.0 * delta * col
        B[update_idx] -= 2.0 * delta * A[update_idx]

        coef[update_idx] = update_value

        if save_history:
            history[iter] = np.sum((X @ coef - y) ** 2) + np.sum(pen_H(coef, Lambda, M))

    if not converged:
        print('Main model did not converge')
    
    return (
        coef,
        history,
    )


class MainModel:
    def __init__(
        self,
        Lambda=100,
        M=10000,
        tol=1e-4,
        maxiter=1000,
        single_tol=1e-6,
        single_maxiter=50,
        single_tol_grad=1e-8,
        save_history=False,
        use_safe=False,
        use_hard_thresholding=False,
        threshold=1e-4,
        init=None,
        coord_peek_strategy="max",
        coord_peek_strategy_k=5,
    ):
        self.Lambda = Lambda
        self.M = M
        self.tol = tol
        self.maxiter = maxiter
        self.single_tol = single_tol
        self.single_maxiter = single_maxiter
        self.single_tol_grad = single_tol_grad
        self.save_history = save_history
        self.use_safe = use_safe
        self.use_hard_thresholding = use_hard_thresholding
        self.threshold = threshold
        self.initial_coef = init
        self.history = None
        self.coord_peek_strategy = coord_peek_strategy
        self.coord_peek_strategy_k = coord_peek_strategy_k

    def fit(self, X, y, **kwargs):
        self.coef_, self.history = fit_main_model(
            X,
            y,
            self.Lambda,
            self.M,
            self.tol,
            self.maxiter,
            self.single_tol,
            self.single_maxiter,
            self.single_tol_grad,
            self.save_history,
            self.use_safe,
            self.initial_coef,
            self.coord_peek_strategy,
            self.coord_peek_strategy_k,
        )
        if self.use_hard_thresholding:
            self.coef_ = np.where(np.abs(self.coef_) > self.threshold, self.coef_, 0)

    def predict(self, X):
        return X @ self.coef_

class MainModelNoiseless(MainModel):
    def fit(self, X, theta_true, **kwargs):
        super().fit(X, X @ theta_true)