🛡️ csPCa Risk Assistant (ROI-Targeted)
A Meta-stacking Ensemble and Uncertainty-Aware Framework for Clinical Decision Support

1. Overview

csPCa Risk Assistant is a clinical decision support tool designed to estimate the probability of Clinically Significant Prostate Cancer (csPCa)—defined as ISUP Grade Group ≥ 2. 

Unlike conventional prostate cancer risk calculators, this system is specifically designed
for ROI-only MRI-targeted biopsy scenarios and excludes systematic biopsy assumptions,
thereby optimizing decision support for contemporary MRI-first pathways.

The model utilizes a Stacking Ensemble Architecture to integrate clinical variables and imaging data, providing not only a point estimate of risk but also a quantified uncertainty analysis using Bootstrap resampling.

By incorporating evolutionary optimization, this framework moves beyond static risk calculators and adaptively combines predictive models under clinical constraints.

2. Clinical Parameters

The assistant processes the following standardized inputs:

Age: Validated for 55 – 75 years.

PSA Dynamics: Total PSA (ng/mL) and calculated PSA Density (PSAD).

Imaging: PI-RADS Max Score (≥ 3).

Prostate Volume: Measured via MRI (mL).

Clinical History: DRE findings, Family History, and prior Biopsy status.

3. Key Features

Advanced Nonlinear Modeling: Uses Cubic Splines (B-splines) for log-transformed PSA data to capture complex biological signals.

Dynamic Calibration: Features a logit-offset calibration module based on the PRECISION Trial standards, allowing local prevalence adjustment.

Uncertainty Quantification: Generates 95% Confidence Intervals (CI) via 1,000 bootstrap iterations to assist in risk-benefit discussions.

Clinician-Centric UI: Built with Streamlit for a high-performance, intuitive workflow.

4. Technical Architecture

The system employs a multi-layer stacking approach:

Base Layer: Diverse machine learning estimators trained on high-quality single-center clinical data.

Meta-Learner: A regularized logistic regression that optimally weights base predictions.

Meta-Stacking Layer: An evolutionary optimization layer based on Differential Evolution (DE)
is applied to refine ensemble composition and weighting,
optimizing performance under clinically constrained objective functions.

Bootstrap Layer: Uncertainty estimation engine for robust risk distribution mapping.

The specific feature engineering pipeline, spline knot placement, ensemble hyperparameters,
and calibration coefficients are intentionally not disclosed and remain proprietary.

5. Scalability & Generalizability 

While the current iteration is optimized based on a high-volume single-center cohort, the modular architecture allows for seamless integration of multi-center data. The inclusion of a Logit-offset Calibration module ensures the tool remains clinically relevant across different geographical regions with varying csPCa prevalence.

6. Intended Clinical Scope

Intended Use:
This tool is designed to support shared decision-making in men with PI-RADS ≥ 3 lesions
undergoing MRI-targeted biopsy and is not validated for screening or systematic biopsy pathways.

7. Intellectual Property & Compliance

Copyright: © 2026 Tran Trung Thanh. All rights reserved.

Methodology: This system integrates a meta-stacking ensemble optimized via Differential Evolution, a Bootstrap-based Uncertainty Engine, and a Logit-offset Calibration module aligned with the PRECISION Trial benchmark.

Registry: This work is part of an internally registered clinical decision support research framework.

How to Cite: Tran, T. T. (2026). csPCa Risk Assistant: A Meta-stacking Ensemble and Uncertainty-Aware Framework for ROI-Targeted Prostate Biopsy Decision Support. GitHub Repository.

Disclaimer: For Research and Educational purposes. Always prioritize clinical findings (DRE, PSAD) in conjunction with these estimates. This tool is not a certified medical device and should not be used as a sole basis for clinical decisions.
