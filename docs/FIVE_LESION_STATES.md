# Five lesion states

The reporting taxonomy contains five mutually exclusive states:

- `no_core_fracture`
- `lateral_malleolus_fracture`
- `medial_malleolus_fracture`
- `posterior_malleolus_fracture`
- `distal_tibial_fracture`

The taxonomy is applied after the four detector scores have been produced. Its
priority order is explicit in `algorithm/schemas/five_lesion_states.yaml`.
It does not convert the detector into a five-class model and does not replace
the four-site localization outputs.
