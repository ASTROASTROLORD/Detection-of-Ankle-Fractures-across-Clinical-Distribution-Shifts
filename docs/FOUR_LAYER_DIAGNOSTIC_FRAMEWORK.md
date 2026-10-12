# Four-layer diagnostic framework

```text
patient fracture recognition
             |
             v
lesion localization
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

With image-grouped annotations, all lesion quantities use the unified
`five_lesion_states.py` matching result. Layer 2 reports the number localized;
layer 3 builds the matched-lesion site matrix; layer 4 splits correct matches
by the frozen class thresholds. The five outcome counts partition all reference
lesions, including unmatched ones. Patient recognition and the original site
summary continue to use patient maximum scores.

The complete `examples/patient_input.json` demonstrates this integration.
See [FIVE_LESION_STATES.md](FIVE_LESION_STATES.md).
