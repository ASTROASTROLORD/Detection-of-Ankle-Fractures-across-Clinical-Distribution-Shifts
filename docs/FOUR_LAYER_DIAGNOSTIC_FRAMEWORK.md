# Four-layer diagnostic framework

```text
patient fracture recognition
             |
             v
same-class lesion detection
             |
             v
post-localization fracture-site typing
             |
             v
fixed-threshold detection performance
```

The framework keeps patient-level recognition, lesion-level localization, and
fracture-site typing as separate quantities. A change in one layer must not be
interpreted as a change in another layer. FROC and IoU analyses identify
whether an observed loss is associated with false positives, localization
tolerance, or fracture-site confusion.

Level 2 uses per-image, same-class matching at IoU >=0.50. It pools prediction
scores into a FROC curve and reports the highest observed sensitivity within
0.125, 0.25, 0.5, 1 and 2 false positives per radiograph, without interpolation.
`evaluate_lesion_detection(images)` accepts all images of a patient or cohort.
For cohort results, supply the complete cohort including negative images.
The patient pipeline computes these quantities over that patient's images.

Layers 3 and 4 share the class-agnostic matches from `five_lesion_states.py`.
Layer 3 builds the conditional site matrix; layer 4 splits correct matches by
the frozen class thresholds. The five outcome counts partition all reference
lesions. Patient recognition uses patient maximum scores.

The existing output key `layer_2_lesion_localization` is retained. Its legacy
`localized_lesions` and `localization_sensitivity` fields now follow same-class
detection, as do `detected_lesions` and `detection_sensitivity`. These summarize
all retained predictions down to score 0.05. Budget-specific FROC results are
in `sensitivity_at_fp_per_radiograph`.

The complete `examples/patient_input.json` demonstrates this integration.
See [FIVE_LESION_STATES.md](FIVE_LESION_STATES.md).
