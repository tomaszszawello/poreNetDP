# %%
import numpy as np
import matplotlib.pyplot as plt


# ---------- Analytical functions ----------

def q_high(alpha, beta):
    """
    Threshold flow ratio q_high(alpha, beta) above which
    the interface sits fully in the bottom layer.

    alpha : float or array
        Relative thickness of the middle layer: alpha = d / D.
    beta : float
        Permeability ratio beta = k1 / k0.

    Returns
    -------
    q_high : float or array
    """
    alpha = np.asarray(alpha)
    return ((1 - alpha) + 2 * beta * alpha) / (1 - alpha)


def q_low(alpha, beta):
    """
    Threshold flow ratio q_low(alpha, beta) below which
    the interface sits fully in the top layer.

    alpha : float or array
        Relative thickness of the middle layer: alpha = d / D.
    beta : float
        Permeability ratio beta = k1 / k0.

    Returns
    -------
    q_low : float or array
    """
    alpha = np.asarray(alpha)
    return (1 - alpha) / ((1 - alpha) + 2 * beta * alpha)


def eta_piecewise(alpha, beta, q):
    """
    Compute dimensionless interface position eta = y*/D
    for a given middle-layer fraction alpha, permeability ratio beta
    and flow ratio q = Qt / Qb.

    Uses a 3-region piecewise expression, depending on q relative
    to q_low(alpha, beta) and q_high(alpha, beta):

      - Region A (q >= q_high): interface in bottom layer
      - Region B (q_low <= q <= q_high): interface in middle layer
      - Region C (q <= q_low): interface in top layer

    Parameters
    ----------
    alpha : array-like
        Middle-layer fraction(s) d/D.
    beta : float
        Permeability ratio k1/k0.
    q : float
        Flow ratio Qt/Qb (top/bottom).

    Returns
    -------
    eta : ndarray
        Same shape as alpha.
    """
    alpha = np.asarray(alpha)
    res = np.empty_like(alpha, dtype=float)

    for i, a in enumerate(alpha):
        qh = q_high(a, beta)
        ql = q_low(a, beta)

        if q >= qh:
            # Case A: interface in bottom layer
            res[i] = ((1 - a) + beta * a) / (q + 1)

        elif q <= ql:
            # Case C: interface in top layer
            res[i] = (1 + q * a * (1 - beta)) / (q + 1)

        else:
            # Case B: interface in middle layer
            num = (-a * beta * q
                   + a * beta
                   + a * q
                   - a
                   + beta * q
                   + beta
                   - q
                   + 1)
            den = 2 * beta * (q + 1)
            res[i] = num / den

    return res


def eta_piecewise_q(alpha, beta, q_array):
    """
    Same as eta_piecewise, but with a possibly different q for each alpha.

    Parameters
    ----------
    alpha : array-like
        Middle-layer fractions d/D.
    beta : float
        Permeability ratio k1/k0.
    q_array : array-like
        Flow ratios Qt/Qb, same shape as alpha.

    Returns
    -------
    eta : ndarray
        Interface position for each (alpha, q_i).
    """
    alpha = np.asarray(alpha)
    q_array = np.asarray(q_array)
    res = np.empty_like(alpha, dtype=float)

    for i, a in enumerate(alpha):
        q = q_array[i]
        qh = q_high(a, beta)
        ql = q_low(a, beta)

        if q >= qh:
            # Case A: interface in bottom layer
            res[i] = ((1 - a) + beta * a) / (q + 1)

        elif q <= ql:
            # Case C: interface in top layer
            res[i] = (1 + q * a * (1 - beta)) / (q + 1)

        else:
            # Case B: interface in middle layer
            num = (-a * beta * q
                   + a * beta
                   + a * q
                   - a
                   + beta * q
                   + beta
                   - q
                   + 1)
            den = 2 * beta * (q + 1)
            res[i] = num / den

    return res


# ---------- Parameters and helper arrays ----------

# alpha range: middle-layer thickness fraction d/D
alphas = np.linspace(0.01, 0.99, 400)

# permeability ratios beta = k1/k0
betas = [0.01, 0.1, 0.5]

# injection ratios q = Qt/Qb (single value here)
qs = [1.1]


# ---------- Figure 1: eta vs alpha for fixed q, various betas ----------

fig, axes = plt.subplots(1, len(qs), figsize=(5 * len(qs), 4), sharey=True)

# If only one subplot, make axes iterable
if len(qs) == 1:
    axes = [axes]

for ax, q in zip(axes, qs):
    # dashed line showing middle–top/bottom layer border at eta = (1 - alpha)/2
    ax.plot(alphas, (1 - alphas) / 2, 'k--', label='layer border')

    # curves for different beta
    for beta in betas:
        eta_vals = eta_piecewise(alphas, beta, q)
        ax.plot(alphas, eta_vals, label=fr"$\beta={beta}$")

    ax.set_title(fr"$q = {q}$")
    ax.set_xlabel(r"$\alpha = d/D$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel(r"$\eta = y^*/D$")
fig.tight_layout()
plt.show()


# ---------- Figure 2: eta vs alpha for piecewise-constant q(alpha) ----------

# Define a q profile that jumps as alpha increases:
#   alpha ∈ [0, 0.5)       -> q0
#   alpha ∈ [0.5, 0.75)    -> q1
#   alpha ∈ [0.75, 0.875)  -> q2
#   alpha ∈ [0.875, 1]     -> q3
q0, q1, q2, q3 = 1.1, 1.2, 1.3, 1.4
betas_q = [0.01]  # choose a single beta here

# build q(alpha) with segments
n = len(alphas)
q_array = np.empty(n, dtype=float)

# indices for the 4 segments
i0 = int(0.5 * n)
i1 = int(0.75 * n)
i2 = int(0.875 * n)

q_array[:i0]      = q0
q_array[i0:i1]    = q1
q_array[i1:i2]    = q2
q_array[i2:]      = q3

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

# layer border
ax.plot(alphas, (1 - alphas) / 2, 'k--', label='layer border')

# vertical lines marking where q changes
ax.vlines([alphas[i0], alphas[i1], alphas[i2]],
          0, 0.5, colors='r', linestyles='dotted', label=r'$q$ change')

# interface curve for selected beta and piecewise q(alpha)
for beta in betas_q:
    eta_vals = eta_piecewise_q(alphas, beta, q_array)
    ax.plot(alphas, eta_vals, label=fr"$\beta={beta}$")

ax.set_title(
    r"Interface position vs middle band width" "\n"
    r"for increasing $q$: $1.1 \rightarrow 1.2 \rightarrow 1.3 \rightarrow 1.4$"
)
ax.set_xlabel(r"$\alpha = d/D$")
ax.set_ylabel(r"$\eta = y^*/D$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='lower left')

fig.tight_layout()
plt.show()
# %%
