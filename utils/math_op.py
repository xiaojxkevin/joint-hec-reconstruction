import numpy as np
from scipy.optimize import linprog


def inv(T: np.ndarray):
    """
    Inverse a SE(3) matrix
    """
    assert T.shape == (4, 4), f"Wrong shape: {T.shape}, should be (4, 4)"
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def skew(v: np.ndarray):
    """
    Skew-symmetric matrix
    """
    assert v.shape == (3, 1), f"Wrong shape: {v.shape}, should be (3, 1)"
    return np.array(
        [[0, -v[2, 0], v[1, 0]], [v[2, 0], 0, -v[0, 0]], [-v[1, 0], v[0, 0], 0]],
        dtype=np.float64,
    )


def minimize_L1_norm(A: np.ndarray, b: np.ndarray):
    """
    Minimizes the L1 norm of (b - A*x) using linear programming.

    We introduce auxiliary variables t such that:
        t_i >= |b_i - (A*x)_i|  for i = 1, ..., m,
    and minimize sum_i t_i.

    This is formulated as:
      minimize      sum_{i=1}^m t_i
      subject to    A*x - t <=  b
                    -A*x - t <= -b
                    t >= 0
    where x ∈ ℝⁿ and t ∈ ℝᵐ.

    Parameters:
      A : (m, n) numpy array
      b : (m,) numpy array

    Returns:
      x_opt : optimal x vector (n,)
      t_opt : optimal auxiliary variables (m,)
      obj_val : minimum L1 norm value
    """
    m, n = A.shape

    # Decision variables: [x_0, ..., x_{n-1}, t_0, ..., t_{m-1}]
    # Objective: minimize sum(t_i) = [0, ..., 0, 1, ..., 1]
    c = np.hstack([np.zeros(n), np.ones(m)])

    # Constraint 1: A*x - t <= b
    A2 = np.hstack([A, -np.eye(m)])
    b2 = b

    # Constraint 2: -A*x - t <= -b
    A1 = np.hstack([-A, -np.eye(m)])
    b1 = -b

    # Combine constraints
    A_ub = np.vstack([A1, A2])
    b_ub = np.concatenate([b1, b2])

    # Define bounds for x (unbounded) and t (non-negative)
    bounds = [(None, None)] * n + [(0, None)] * m

    # Solve the linear program using the HiGHS method
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.success:
        x_opt = res.x[:n]
        t_opt = res.x[n:]
        return x_opt, t_opt, res.fun
    else:
        raise ValueError("Linear programming failed: " + res.message)
