# Model card

## Intended use

Research and offline analysis of ankle radiographs, including patient-level
fracture recognition, lesion localization, fracture-site typing, and error
analysis.

## Out of scope

The algorithm is not a clinical diagnostic device, triage system, treatment
recommendation system, or autonomous medical decision-maker. It requires
appropriate clinical oversight and independent validation before any regulated
use.

## Outputs

The model produces four fracture-site detection outputs. Patient-level scores,
fixed-threshold decisions, FROC values, IoU analyses, original patient site
summaries and five reference-lesion outcomes are derived outputs. Lesion
outcome attribution requires reference annotations and uses per-image matches;
it cannot be generated as an error label for an unannotated radiograph.
