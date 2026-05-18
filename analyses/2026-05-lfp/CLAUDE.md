# Multi-Channel Neuropixel LFP Pre-Processing Pipeline

## Context
I'm working on optimizing multi-channel pre-processing of low-field potential (LFP) recordings from Neuropixel probes, the goal being recovering the current source density estimation. CSD is inherently very noisy (double spatial diff). I have an existing pipeline but need to explore parameter tuning and aggressive noise attenuation techniques. I want the techniques to work for both the NP1 channel layout in checkerboard pattern and the NP2 channel layout that is arranged in straight columns. 

## Existing Codebase
- **Data loading script**: `analyses/2026-05-lfp/2025-05-NP1NP2.py`
  - Contains experiments comparing Neuropixel 1 and 2 probes
- **Core DSP library**: Package `ibldsp`, `uv pip install ibl-neuropixel`
  - `ibldsp.voltage` - voltage preprocessing functions
  - `ibldsp.fourier` - Fourier transform utilities
  - `ibldsp.cadzow` - Cadzow denoising implementation

## Objectives
1. **Parameter search framework**: Design systematic exploration of preprocessing parameters (filter cutoffs, window sizes, denoising thresholds, etc.)
2. **Aggressive multi-channel noise attenuation**: Suggest and implement advanced techniques such as:
   - Spatial filtering exploiting probe geometry
   - Multi-taper methods
   - Robust covariance estimation
   - Adaptive filtering approaches
   - Subspace-based denoising beyond Cadzow

## Tasks
1. Review the existing data loading and preprocessing pipeline
2. Identify current parameters and their ranges
3. Suggest 3-5 advanced multi-channel noise reduction techniques appropriate for Neuropixel LFP data
4. Implement a modular framework for testing different preprocessing combinations
5. Create evaluation metrics for comparing preprocessing approaches (SNR improvement, artifact reduction, signal preservation). There is already a nice visualisation technique in `2026-03-24_NP1NP2.py` that uses cross-correlation between simultaneously recorded NP1 and NP2 channels.
6. Generate visualization tools for comparing results across parameter sets

## Deliverables
- Analysis of current pipeline
- New noise attenuation functions integrated with existing `ibldsp` library structure
- Comparison framework with metrics and visualizations
- Recommendations based on systematic evaluation and cross-correlation peaks

## Notes
- Preserve multi-channel spatial structure (Neuropixel probe geometry for NP1 and NP2 are different and should be taken into account)
- Consider computational efficiency for large-scale processing
- Maintain compatibility with existing `ibldsp` API conventions

