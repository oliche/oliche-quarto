# Multi-Channel Neuropixel LFP Pre-Processing Pipeline

## Context

in addition to our atlas it would also be useful to supply the community with a set of landmarks - specific places in the brain where there is a sharp transition in predicted features that scientists can use to help determine their location(the DG-thalamus LFP landmark from the repro-ephys paper is a nice example of this.)

we could search for more such landmarks by brute force: for each region boundary (could start with cosmos and if it works try beryl next): find all penetrations that go through this region boundary  for each ephys feature: make a plot like fig 3b of the repro-ephys paper (https://elifesciences.org/articles/100840#fig3) run a t-test (or similar) to see if there is a sharp transition across the boundary
if yes (and there are enough penetrations and the effect size is large enough) then add this to the list of landmarks to report (or sort the landmarks by p-value)

and then collect the resulting features into a collection of supplemental figures that we could check and then add to the paper to document our claims about these landmarks.

we could also check the results against some known cases - eg the void-cortex transition, DG-thalamus, white matter to gray matter, ventricle to non-ventricle, etc. we can also do some data splitting / xvalidation to further check the results + make sure the suggested landmarks are real + useful.

## Plan

1. Load the ephys-atlas channel features using this skill: /home/olivier/PycharmProjects/EphysAtlas/paper-ephys-atlas/.claude/skills/load-channel-features-dataframe.md 
2. At the Cosmos level, create a matrix of the counts of the number of transitions across the region boundaries (should be a 13x13 matrix). Channels are first aggregated by depth (axial_um) within each probe — taking the modal Cosmos_id across channels at the same depth — before transitions between adjacent depth levels are detected. The matrix is directional (A→B ≠ B→A).
3. Display the matrix as a heatmap, burn the diagonal and choose the dynamic color range optimized for the off-diagonal terms
4. Look for sharp transitions for each the boundaries in the matrix, above a certain threshold to determine
5. Document the findings in displays



