# Input and output schema

All boxes use image-space `[x1, y1, x2, y2]` coordinates. Labels are integer
indices 0–3; confidence scores lie in [0, 1].

## Unified five-state interface

`interface.summarize_patient_diagnosis` accepts existing patient output with
four `class_scores`, or an `images` array with explicit reference annotations.
Each image has a unique `image_id`, optional `scene`, `detections` and
`ground_truth`. Reference items contain `gt_id`, `label` and `box`; detections
contain `label`, `score` and `box`. Reference IDs are unique within an image.

For patient input, the output preserves existing fields and includes
`patient_site_summary` and the original `lesion_state` reporting field.
For annotated images, it provides `method`, `summary` and `lesions`; optional
explicit patient scores add `patient_site_summary` to that result.
Proportions use all reference lesions as denominator. Empty reference sets
have zero counts and JSON `null` proportions.

Contracts: `five_lesion_states_input.json`, `five_lesion_states_output.json`
and the unit-specific definitions in `five_lesion_states.yaml`.

## Patient inference interface

`interface.infer_patient` requires `patient_id` and accepts either an `images`
array grouping detections by image, or the original `image_ids` array plus flat
`detections` with image identifiers. Supply references inside each image or
use `ground_truth_by_image`. Every image must declare its references, including
an empty array for a negative image. The original `ground_truth_by_class`
format applies to one image only. If no explicit patient targets are supplied,
image-grouped reference labels determine those targets.

Output includes `class_scores`, `class_predictions`, `layers`, the four layer
fields, `patient_site_summary` and `lesion_state`. With image-grouped reference
annotations it additionally includes `five_lesion_analysis`, using the same
matched rows as layers 2–4. Only annotated inputs enable lesion analysis.

Contracts: `patient_input.json` and `patient_output.json`. Complete synthetic
examples are under `examples/`. Schemas are installed as package resources.
