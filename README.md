# H → ZZ* → 4ℓ: reconstruction chain and detector-response study

A pure-Python (uproot + awkward + numpy) reconstruction of Higgs boson decays to four leptons in CMS 2012 simulation. The data file contains simulated **signal only** (m_H = 125 GeV): there is no background and no real collision data, so this project does **not** claim a Higgs discovery. It studies what the detector does to the signal.

## Method
1. Select good muons and electrons (pT, η, isolation, impact-parameter cuts).
2. Keep events with exactly 4 good leptons (39,981 of 299,973 events).
3. Find the best opposite-sign, same-flavour pairing; Z1 is the pair closest to the Z mass.
4. Apply the event selection and record a cutflow.
5. Fit the four-lepton mass peak in each channel (4μ, 4e, 2e2μ) with a Crystal Ball function.
6. Measure how the resolution changes with the number of leptons in the detector endcap.

## Results
![m4l by channel](results/m4l_channels_fit.png)

| Channel | Peak [GeV] | σ (Crystal Ball) [GeV] | Share of events in file |
|---|---|---|---|
| 4μ | 124.82 ± 0.02 | 1.23 ± 0.01 | 3.53 % |
| 4e | 125.29 ± 0.06 | 2.94 ± 0.05 | 2.39 % |
| 2e2μ | 125.14 ± 0.03 | 2.12 ± 0.02 | 5.21 % |

- Resolution is best in the muon channel and worst in the electron channel.
- It degrades when leptons are in the endcap (|η| > 1.5): for 4e, σ68 rises from 2.96 GeV (no endcap leptons) to 4.45 GeV (two or more).
- Full numbers: [`results/summary.txt`](results/summary.txt). Other plots: `results/cutflow.png`, `results/z1_vs_z2.png`.

## Limitations
- The "share of events" is relative to the events stored in this reduced file, not the number of generated events, so it is not an absolute acceptance × efficiency.
- The selection cuts approximate, but are not identical to, the official CMS analysis.
- The Crystal Ball fit describes the 4e channel only approximately.
- To turn this into a real rediscovery, run the same steps on the ZZ → 4ℓ background simulations and the 2012 DoubleMuParked / DoubleElectron data, then compare data with background plus signal.

## Run
```bash
cd higgs-4l-signal-study
python higgs4l_signal_study.py
```
Outputs go to `results/`. See `data/README.md` for the data file.

## Data and citation
- Wunsch, S. (2021). *SMHiggsToZZTo4L dataset in reduced NanoAOD format for education and outreach*. CERN Open Data Portal. DOI: [10.7483/OPENDATA.CMS.8FLU.UIQJ](https://doi.org/10.7483/OPENDATA.CMS.8FLU.UIQJ)
- Related official analysis (not reused here): Wunsch, S. (2021). *Analysis of Higgs boson decays to four leptons ... using ROOT's RDataFrame*. DOI: [10.7483/OPENDATA.CMS.F7HD.P3K4](https://doi.org/10.7483/OPENDATA.CMS.F7HD.P3K4)
