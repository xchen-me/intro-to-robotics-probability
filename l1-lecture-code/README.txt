Lecture 1: sums of Uniform[0,1] random variables

Author: Xu Chen. The experiment covers n=1,2,3,12.

Setup: 100,000 trials of IID Uniform[0,1] variables. Each row is one trial.
Assumptions: independent and identical draws, mean 1/2, variance 1/12.
Takeaway: raw sums broaden, while centered and scaled sums approach N(0,1).
Two uniforms have an exact triangular sum density. Finite sums are not exactly
Gaussian. Histogram bars represent density, normalized by the actual bin widths.

Python 3.11+, macOS/Linux, from modernized_lectures:
  python3 -m venv .venv-l1
  .venv-l1/bin/python -m pip install -r code/l1/requirements.txt
  .venv-l1/bin/python code/l1/clt_uniform.py
Windows PowerShell:
  py -3 -m venv .venv-l1
  .venv-l1\Scripts\python.exe -m pip install -r code/l1/requirements.txt
  .venv-l1\Scripts\python.exe code/l1/clt_uniform.py
Add --no-show to save the results without opening plot windows.

MATLAB R2016b+, from modernized_lectures as Current Folder:
  addpath('code/l1');
  results = clt_uniform_demo;
Uses base MATLAB functions, with no Statistics Toolbox requirement.
Identical seeds in different languages do not imply identical random samples.

For n=1,2,3,12, expected means are 0.5,1,1.5,6 and variances are
1/12,1/6,1/4,1. The standardized variables have mean zero and variance one.
Both programs save results to code-results/l1. Full annotated listings are also
embedded in the L1 HTML/PDF lecture notes.

Two more standalone demos live alongside clt_uniform.py:
  range_measurement_demo.py --sigma 0.02   (the 30 cm wall sensor)
  summed_disturbances_demo.py              (CLT on a bimodal mixture)
Both take --no-show and save to code-results/l1 the same way. All code in
this folder is also mirrored at
github.com/xchen-me/intro-to-robotics-probability/tree/main/l1-lecture-code.

Figures and camera layout
The slides' technical artwork is editable in figures/l1/*.tex. Rebuild figures
with the build environment: python support/l1_figures.py, after the Python demo.
Rebuild all L1 formats with: python support/build.py L1_probability_foundations
The slides reserve an empty 16:9 camera area at bottom right (18% slide width).
Slide titles, body text, equations and the loot table remain editable in PPTX;
TikZ drawings are embedded images and are edited through the supplied TeX files.
