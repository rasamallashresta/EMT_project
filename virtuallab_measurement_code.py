"""
03_virtual_measurement.py
Software replacement for the signal generator + spectrum analyzer experiment.

Takes a "true" SE curve (CST result or analytic) and emulates what a real bench would record:
  - finite spectrum-analyzer noise floor and dynamic range
  - amplitude uncertainty (cable/connector/antenna repeatability, +/- dB)
  - antenna / box positioning error that slightly shifts resonance frequencies
  - random seam-leakage penalty (imperfect screw / copper-tape contact)
Runs N Monte Carlo "repeated measurements" and plots mean +/- 1 sigma against the true curve.

USAGE   (args: csv file, column number, name tag for the figure)
  python 03_virtual_measurement.py results/analytic_slot.csv 1 slot1   # column 1 = SE of one slot
  python 03_virtual_measurement.py results/cst_SE.csv 1               # column 1 of CST file
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

P_TX_DBM = 0.0           # signal generator output
PATH_LOSS_DB = 40.0      # antenna-to-antenna loss without the box (reference measurement)
NOISE_FLOOR_DBM = -90.0  # analyzer displayed average noise level with chosen RBW
AMP_SIGMA_DB = 0.7       # random amplitude uncertainty (1 sigma)
FREQ_JITTER = 0.005      # +/-0.5 % shift from position/box tolerance
SEAM_MEAN_DB = 3.0       # average SE loss from imperfect seam contact
SEAM_SIGMA_DB = 2.0
N_RUNS = 200


def dbm_add(*levels):
    return 10 * np.log10(sum(10 ** (x / 10) for x in levels))


def measure(f, se_true):
    # frequency shift: SE(f * (1+eps))
    eps = rng.uniform(-FREQ_JITTER, FREQ_JITTER)
    se = np.interp(f * (1 + eps), f, se_true)
    se = se - np.clip(rng.normal(SEAM_MEAN_DB, SEAM_SIGMA_DB), 0, None)  # seam leakage penalty
    p_ref = P_TX_DBM - PATH_LOSS_DB + rng.normal(0, AMP_SIGMA_DB, f.size)
    p_box = P_TX_DBM - PATH_LOSS_DB - se + rng.normal(0, AMP_SIGMA_DB, f.size)
    p_box_meas = dbm_add(p_box, np.full(f.size, NOISE_FLOOR_DBM))        # noise floor limits what is seen
    return p_ref - p_box_meas


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/analytic_slot.csv"
    col = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    tag = sys.argv[3] if len(sys.argv) > 3 else "default"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    f, se_true = data[:, 0], data[:, col]
    runs = np.array([measure(f, se_true) for _ in range(N_RUNS)])
    mean, std = runs.mean(0), runs.std(0)

    dyn = P_TX_DBM - PATH_LOSS_DB - NOISE_FLOOR_DBM
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.fill_between(f / 1e9, mean - std, mean + std, alpha=.3, label="Virtual measurement +/- 1 sigma")
    plt.plot(f / 1e9, mean, label="Virtual measurement mean")
    plt.plot(f / 1e9, se_true, "k--", label="True (simulated) SE")
    plt.axhline(dyn, color="r", ls=":", label=f"Dynamic-range limit ({dyn:.0f} dB)")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("SE (dB)")
    plt.title(f"Virtual bench measurement, {N_RUNS} runs")
    plt.ylim(-5, min(dyn + 15, 80)); plt.grid(True, alpha=.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"results/fig7_virtual_{tag}.png", dpi=140)

    err = np.abs(mean - se_true)
    print(f"Dynamic range of virtual setup: {dyn:.0f} dB")
    print(f"Mean |measured - true| = {err[se_true < dyn - 10].mean():.2f} dB (below dynamic-range limit)")
    print(f"Wrote results/fig7_virtual_{tag}.png")
