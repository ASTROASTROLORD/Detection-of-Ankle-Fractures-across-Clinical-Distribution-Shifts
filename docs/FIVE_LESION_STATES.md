# Unified five-state analysis

## Unit and candidate set

The unit is one annotated reference lesion. Every reference lesion receives
exactly one outcome. All reference lesions in the selected images form the
denominator; predictions on fracture-negative images do not create reference
lesions. Report false-positive burden separately with FROC.

Retain at most 100 predictions per image, sorted by descending score, with
scores >=0.05. For equal scores, first group predictions by ascending detector
class index, preserving input order within a class, and then stable-sort by
score. Group references by class in the same way. This matches the final FROC
evaluator's ordering. Match independently within each image; coordinates from
different images must never be pooled.

For each prediction, select the highest-IoU reference among references not yet
matched, irrespective of fracture class. Accept IoU >=0.50 and consume that
reference. Each prediction and reference can participate in at most one match.

## Five mutually exclusive outcomes

| Outcome ID | Display label | Rule |
| --- | --- | --- |
| `no_effective_candidate` | No effective localization candidate | No match and maximum retained-candidate IoU <0.25, including no retained candidates |
| `localization_insufficient` | Localization insufficient | No match, but maximum retained-candidate IoU >=0.25 |
| `wrong_class` | Incorrect class | Matched prediction's fracture site differs from reference site |
| `correct_below_frozen_threshold` | Correct, below threshold | Match and site are correct, but matched score is below the reference-site threshold |
| `correct_frozen_threshold_met` | Correct, threshold met | Match and site are correct, and matched score reaches the reference-site threshold |

Localization insufficient includes an otherwise overlapping box consumed by
another lesion during one-to-one matching. Wrong class means the **matched
box** is incorrectly typed; another correct-class candidate does not override
that outcome. Class thresholds do not filter the matching candidate set.
Boundary equality is accepted at score 0.05, candidate IoU 0.25, localization
IoU 0.50, and each class-specific decision threshold.

The default frozen thresholds, in detector class order, are:

| Class | Threshold |
| --- | ---: |
| Lateral malleolus | 0.63 |
| Medial malleolus | 0.76 |
| Posterior malleolus | 0.79 |
| Distal tibia | 0.68 |

Supply other declared thresholds through the API or CLI only when evaluating
a different operating point. The output always records the values actually used.
Scores below the archive floor are excluded. No effective candidate therefore
means absence of an effective **retained** candidate, not proof that the model
never generated any lower-scoring box.

## Input, output and command line

The JSON input contains `images`. Each image contains a unique `image_id`, an
optional `scene` (default `unspecified`), `ground_truth`, and `detections`.
Reference items contain `gt_id`, `label` and `box`; detection items contain
`label`, `score` and `box`. Labels are integers 0–3. Boxes use image-space
`[x1, y1, x2, y2]` coordinates, not COCO `[x, y, width, height]` coordinates.
Reference IDs must be unique within an image. Image IDs must be unique in the
input, even across scenes.

```bash
python -m interface.summarize_patient_diagnosis \
  examples/five_lesion_states_input.json lesion_analysis.json \
  --threshold 0.63 0.76 0.79 0.68
```

Output fields:

- `method`: matching definition, score/IoU gates, class thresholds and image count.
- `summary.all` and `summary.by_scene`: reference-lesion count, five outcome
  counts and proportions. Empty scenes have zero counts and `null` proportions.
- `lesions`: scene/image/reference IDs, reference class and box, matched
  prediction index/class/box/score/rank/IoU, best candidate IoU and outcome.
  Prediction index refers to the original image detection-array index (zero
  based); prediction rank refers to the retained matching order (one based).
  Unmatched prediction fields are `null`.

The synthetic example contains one reference lesion in each outcome. It does
not contain clinical data or reproduce the article's observed proportions.

## Python API

```python
from algorithm.diagnostic_analysis.five_lesion_states import summarize_diagnosis

result = summarize_diagnosis({"images": images}, thresholds=(0.63, 0.76, 0.79, 0.68))
```

`decompose_image(image_id, predictions, ground_truth, thresholds)` accepts
`Detection` records and returns the per-image lesion rows. `summarize_outcomes`
summarizes those rows. The same `five_lesion_states.py` module
also contains the original patient-level fracture-site reporting functions.

## Patient-level reporting and the four-layer pipeline

The original site states are `no_core_fracture` and the four fracture sites.
The priority is distal tibia, posterior malleolus, medial malleolus, then lateral
malleolus. `no_core_fracture` means no site reached its declared score threshold;
it is not a reference annotation that proves absence of fracture.

`assign_patient_state` names that reporting function explicitly;
`assign_lesion_state` and `summarize_states` retain the original public API.
For existing patient-output inputs, `summarize_diagnosis` preserves the input
fields and adds `patient_site_summary` and the compatible `lesion_state` field.
For an annotated-image input, it returns the lesion method, summary and records.
Providing explicit patient `class_scores` alongside `images` additionally
returns the site summary, without using patient scores to classify lesions.

`run_four_layer_diagnosis` invokes this same module. With
`ground_truth_by_image`, its `five_lesion_analysis` contains the canonical
lesion matches. Layer 2 separately evaluates same-class detection and FROC.
Layer 3 builds its conditional
site-confusion matrix from those matched rows, and layer 4 reports the same
five-outcome counts alongside thresholded detections. `class_predictions` and
`layers` are included as declared by the patient-output schema.

Run the complete synthetic patient input with:

```bash
python -m interface.infer_patient \
  examples/patient_input.json patient_output.json \
  --threshold 0.63 0.76 0.79 0.68
```

The `images` input groups detections and references within each image. The
original `image_ids` plus flat `detections` form is also supported; add
`ground_truth_by_image` to retain image identity for multi-image analysis.
Every image must have an explicit reference array when lesion analysis is
requested. Reference labels are used for patient targets when no explicit
`target_labels_by_image` is supplied. The original `ground_truth_by_class`
format is limited to one image, since pooling boxes from different radiographs
can create invalid matches.

The unified JSON contracts are `five_lesion_states_input.json` and
`five_lesion_states_output.json`. Patient contracts are `patient_input.json`
and `patient_output.json`. All are distributed with the installed package.
The YAML file groups lesion outcomes and patient site states with their
respective units, thresholds and rules.
