# H -> ZZ* -> 4 leptons: reconstruction chain and detector-response study
# Data: CMS 2012 simulation, reduced NanoAOD, SMHiggsToZZTo4L (m_H = 125 GeV)
#       Wunsch, S. (2021), CERN Open Data Portal, DOI:10.7483/OPENDATA.CMS.8FLU.UIQJ
#
# This file is simulated SIGNAL only (no background, no real data), so this is not a discovery.
# The script builds the full reconstruction chain and measures what the detector does to a
# 125 GeV Higgs: selection efficiency, mass resolution and tails, per decay channel.
# Run it from inside this folder, with the data file in data/.

# %% 1. Imports and settings
import os
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
from scipy.stats import crystalball
from scipy.optimize import curve_fit

try:
    import mplhep as hep
    hep.style.use(hep.style.CMS)
except ImportError:
    pass

os.makedirs("results", exist_ok=True)
Z_MASS = 91.1876                    # GeV
MU_MASS, EL_MASS = 0.10566, 0.000511  # GeV
report = []                         # lines collected here are printed and saved at the end

# %% 2. Load only the branches we need
branches = [
    "Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge", "Muon_pfRelIso04_all",
    "Muon_dxy", "Muon_dxyErr", "Muon_dz", "Muon_dzErr",
    "Electron_pt", "Electron_eta", "Electron_phi", "Electron_charge", "Electron_pfRelIso03_all",
    "Electron_dxy", "Electron_dxyErr", "Electron_dz", "Electron_dzErr",
]
ev = uproot.open("data/SMHiggsToZZTo4L.root")["Events"].arrays(branches)
n_total = len(ev)
report.append(f"Events in file: {n_total}")

# %% 3. Select good leptons (quality cuts on each lepton)
# SIP is a proxy for the 3D impact-parameter significance: sqrt((dxy/err)^2 + (dz/err)^2)
mu_sip = np.sqrt((ev.Muon_dxy / ev.Muon_dxyErr) ** 2 + (ev.Muon_dz / ev.Muon_dzErr) ** 2)
el_sip = np.sqrt((ev.Electron_dxy / ev.Electron_dxyErr) ** 2 + (ev.Electron_dz / ev.Electron_dzErr) ** 2)

good_mu = ((ev.Muon_pt > 5) & (abs(ev.Muon_eta) < 2.4) & (ev.Muon_pfRelIso04_all < 0.4)
           & (abs(ev.Muon_dxy) < 0.5) & (abs(ev.Muon_dz) < 1.0) & (mu_sip < 4))
good_el = ((ev.Electron_pt > 7) & (abs(ev.Electron_eta) < 2.5) & (ev.Electron_pfRelIso03_all < 0.4)
           & (abs(ev.Electron_dxy) < 0.5) & (abs(ev.Electron_dz) < 1.0) & (el_sip < 4))

# %% 4. Keep events with exactly 4 good leptons and turn them into plain (n_events, 4) arrays
n_leptons = ak.num(ev.Muon_pt[good_mu]) + ak.num(ev.Electron_pt[good_el])
has_four = ak.to_numpy(n_leptons == 4)

# muons first, then electrons, in each event
pt = ak.concatenate([ev.Muon_pt[good_mu], ev.Electron_pt[good_el]], axis=1)[has_four]
eta = ak.concatenate([ev.Muon_eta[good_mu], ev.Electron_eta[good_el]], axis=1)[has_four]
phi = ak.concatenate([ev.Muon_phi[good_mu], ev.Electron_phi[good_el]], axis=1)[has_four]
charge = ak.concatenate([ev.Muon_charge[good_mu], ev.Electron_charge[good_el]], axis=1)[has_four]
is_muon = ak.concatenate([ev.Muon_pt[good_mu] * 0 + 1, ev.Electron_pt[good_el] * 0], axis=1)[has_four]

pt, eta, phi, charge, is_muon = (ak.to_numpy(x) for x in (pt, eta, phi, charge, is_muon))
order = np.argsort(-pt, axis=1)                      # sort leptons by pT, highest first
pt, eta, phi, charge, is_muon = (np.take_along_axis(x, order, axis=1) for x in (pt, eta, phi, charge, is_muon))
report.append(f"Events with exactly 4 good leptons: {len(pt)}")

# %% 5. Four-vectors of every lepton
mass = np.where(is_muon == 1, MU_MASS, EL_MASS)
px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

m4l = np.sqrt(np.maximum(energy.sum(1) ** 2 - px.sum(1) ** 2 - py.sum(1) ** 2 - pz.sum(1) ** 2, 0))
n_mu = is_muon.sum(axis=1)
channel = np.select([n_mu == 4, n_mu == 0, n_mu == 2], ["4mu", "4e", "2e2mu"], "other")

# %% 6. Find the best Z1/Z2 pairing
# Four leptons can be paired three ways. A valid pairing has two opposite-sign, same-flavour pairs.
# Z1 is the pair whose mass is closest to the Z mass; the best pairing is the one with the best Z1.
pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
z1 = np.full(len(pt), np.nan)
z2 = np.full(len(pt), np.nan)
best_distance = np.full(len(pt), np.inf)

for (a, b), (c, d) in pairings:
    valid = ((charge[:, a] + charge[:, b] == 0) & (is_muon[:, a] == is_muon[:, b]) &
             (charge[:, c] + charge[:, d] == 0) & (is_muon[:, c] == is_muon[:, d]))
    m_ab = np.sqrt(np.maximum((energy[:, a] + energy[:, b]) ** 2 - (px[:, a] + px[:, b]) ** 2
                              - (py[:, a] + py[:, b]) ** 2 - (pz[:, a] + pz[:, b]) ** 2, 0))
    m_cd = np.sqrt(np.maximum((energy[:, c] + energy[:, d]) ** 2 - (px[:, c] + px[:, d]) ** 2
                              - (py[:, c] + py[:, d]) ** 2 - (pz[:, c] + pz[:, d]) ** 2, 0))
    ab_is_z1 = abs(m_ab - Z_MASS) < abs(m_cd - Z_MASS)
    this_z1 = np.where(ab_is_z1, m_ab, m_cd)
    this_z2 = np.where(ab_is_z1, m_cd, m_ab)
    better = valid & (abs(this_z1 - Z_MASS) < best_distance)
    z1[better], z2[better] = this_z1[better], this_z2[better]
    best_distance[better] = abs(this_z1[better] - Z_MASS)

# %% 7. Event selection with a cutflow
has_pairing = ~np.isnan(z1)
cut_pt = (pt[:, 0] > 20) & (pt[:, 1] > 10)
cut_z1 = (z1 > 40) & (z1 < 120)
cut_z2 = (z2 > 12) & (z2 < 120)
cut_m4l = m4l > 70

cutflow = {
    "4 good leptons": np.ones(len(pt), dtype=bool),
    "OS same-flavour pairing": has_pairing,
    "pT(1)>20, pT(2)>10": has_pairing & cut_pt,
    "40 < m(Z1) < 120": has_pairing & cut_pt & cut_z1,
    "12 < m(Z2) < 120": has_pairing & cut_pt & cut_z1 & cut_z2,
    "m4l > 70": has_pairing & cut_pt & cut_z1 & cut_z2 & cut_m4l,
}
selected = cutflow["m4l > 70"]

report.append("\nCutflow (events with 4 good leptons):")
for name, mask in cutflow.items():
    report.append(f"  {name:26s} {mask.sum():7d}")

report.append("\nSelection efficiency by channel (fraction of ALL events stored in the file):")
for ch in ["4mu", "4e", "2e2mu"]:
    n = (selected & (channel == ch)).sum()
    report.append(f"  {ch:6s} {n:6d}   {100 * n / n_total:5.2f} %")
report.append(f"  total  {selected.sum():6d}   {100 * selected.sum() / n_total:5.2f} %")
report.append("  (denominator = events in this reduced file, not generated events: not an absolute acceptance x efficiency)")

# %% 8. Fit the 4-lepton mass peak in each channel with a Crystal Ball function
# Crystal Ball = Gaussian core + power-law tail on the low-mass side (energy lost to radiation).
low, high, n_bins = 105.0, 140.0, 70
crystal_ball = lambda x, norm, beta, m, loc, scale: norm * crystalball.pdf(x, beta, m, loc=loc, scale=scale)

fits = {}
report.append("\nCrystal Ball fit of the m4l peak:")
for ch in ["4mu", "4e", "2e2mu"]:
    in_range = selected & (channel == ch) & (m4l > low) & (m4l < high)
    counts, edges = np.histogram(m4l[in_range], bins=n_bins, range=(low, high))
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    start = [in_range.sum() * width, 1.2, 3.0, 124.5, 1.5]
    limits = ([0, 0.1, 1.01, 115, 0.3], [np.inf, 5, 50, 130, 6])
    errors = np.sqrt(np.maximum(counts, 1))
    popt, pcov = curve_fit(crystal_ball, centres, counts, p0=start, sigma=errors,
                           absolute_sigma=True, bounds=limits)
    perr = np.sqrt(np.diag(pcov))
    fits[ch] = (centres, counts, errors, popt)
    report.append(f"  {ch:6s} peak = {popt[3]:.2f} +/- {perr[3]:.2f} GeV    sigma = {popt[4]:.2f} +/- {perr[4]:.2f} GeV")

# %% 9. Resolution vs detector region
# Is the mass resolution worse when leptons are in the endcap (|eta| > 1.5)?
# sigma68 = half the width of the central 68% interval (robust against tails).
n_endcap = (abs(eta) > 1.5).sum(axis=1)
report.append("\nMass resolution (sigma68) by number of endcap leptons:")
for ch in ["4mu", "4e", "2e2mu"]:
    for label, in_group in [("0 endcap", n_endcap == 0), ("1 endcap", n_endcap == 1), (">=2 endcap", n_endcap >= 2)]:
        sel = selected & (channel == ch) & in_group & (m4l > low) & (m4l < high)
        if sel.sum() < 100:
            continue
        lo68, hi68 = np.percentile(m4l[sel], [15.87, 84.13])
        report.append(f"  {ch:6s} {label:11s} N={sel.sum():5d}   sigma68 = {(hi68 - lo68) / 2:.2f} GeV")

# %% 10. Plots
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, ch in zip(axs, ["4mu", "4e", "2e2mu"]):
    centres, counts, errors, popt = fits[ch]
    x_fine = np.linspace(low, high, 400)
    ax.errorbar(centres, counts, yerr=errors, fmt="ko", ms=3)
    ax.plot(x_fine, crystal_ball(x_fine, *popt), "r-", lw=2)
    ax.set_title(ch)
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]")
    ax.text(0.04, 0.95, f"peak {popt[3]:.2f} GeV\n$\\sigma$ {popt[4]:.2f} GeV", transform=ax.transAxes, va="top")
axs[0].set_ylabel("Simulated events / 0.5 GeV")
fig.savefig("results/m4l_channels_fit.png", dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(cutflow.keys())
ax.barh(names[::-1], [m.sum() for m in cutflow.values()][::-1], color="steelblue")
ax.set_xlabel("Events")
fig.savefig("results/cutflow.png", dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6, 5))
h = ax.hist2d(z1[selected], z2[selected], bins=[40, 40], range=[[40, 120], [0, 80]], cmin=1)
fig.colorbar(h[3], ax=ax)
ax.set_xlabel("m(Z1) [GeV]")
ax.set_ylabel("m(Z2) [GeV]")
fig.savefig("results/z1_vs_z2.png", dpi=200, bbox_inches="tight")

# %% 11. Save the text summary
print("\n".join(report))
with open("results/summary.txt", "w") as f:
    f.write("\n".join(report))
