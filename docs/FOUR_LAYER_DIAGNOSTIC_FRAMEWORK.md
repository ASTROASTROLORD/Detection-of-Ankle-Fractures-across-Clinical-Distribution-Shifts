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
