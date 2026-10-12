# Deep Learning-based Detection of Ankle Fractures across Clinical Distribution Shifts

## A Multicenter Hierarchical Performance Evaluation

This repository contains the article-facing implementation of the final ankle
fracture detection algorithm described in:

> **Deep Learning-based Detection of Ankle Fractures across Clinical
> Distribution Shifts: A Multicenter Hierarchical Performance Evaluation**

The public code exposes one final detector, a Faster R-CNN with a ResNet-50-FPN
backbone, together with the patient-level diagnostic framework used to
interpret its outputs across clinical distribution shifts. The repository is
intended to make the algorithm interface and the hierarchy of reported
performance measures transparent to readers of the article.

The package contains algorithm code and small, synthetic unit-test inputs only.
It does not contain patient images, original annotations, patient identifiers,
model checkpoints, or private project archives.

> **Research use only.** This software is provided for research and offline
> analysis. It is not a clinical diagnostic device and must not be used as an
> autonomous medical decision system.

## What the algorithm provides

The detector predicts four fracture-site categories from ankle radiographs:

- `lateral_malleolus_fracture`
- `medial_malleolus_fracture`
- `posterior_malleolus_fracture`
- `distal_tibial_fracture`

The detector output is interpreted through a four-layer diagnostic framework:

1. **Patient-level fracture recognition** aggregates the maximum class-specific
   detection score across all radiographs belonging to a patient.
2. **Lesion localization** evaluates whether predicted boxes overlap reference
   lesions at a declared IoU threshold.
3. **Post-localization fracture-site typing** separates correct localization
   from fracture-site confusion.
4. **Fixed-threshold detection performance** reports the final operating point
   using fixed class-specific thresholds.

The diagnostic analysis modules provide category-specific FROC, class-agnostic
FROC, IoU sensitivity analysis, localization confusion matrices, and one unified
five-state analysis module. It provides the original patient-level fracture-site
summary and the five mutually exclusive reference-lesion detection outcomes.
Both are derived analyses; the detector retains its four fracture-site outputs.

## Repository layout

```text
ankle-fracture-algorithm/
├── algorithm/
│   ├── detector/
│   │   ├── faster_rcnn_r50_fpn.py       # final detector definition
│   │   ├── detection_output.py          # detector-output validation
│   │   └── preprocessing.py              # geometry-preserving metadata
│   ├── patient_level/
│   │   ├── pipeline.py                   # four-layer composition
│   │   ├── patient_aggregation.py        # image-to-patient aggregation
│   │   ├── threshold_policy.py           # fixed threshold definitions
│   │   └── layer_*                       # four diagnostic layers
│   ├── diagnostic_analysis/              # FROC, IoU and error analyses
│   └── schemas/                          # JSON/YAML input-output contracts
├── interface/
│   ├── infer_image.py                    # normalize image-level output
│   ├── infer_patient.py                  # run four-layer patient analysis
│   └── summarize_patient_diagnosis.py     # unified site and lesion-state summary
├── examples/                             # synthetic lesion and patient inputs
├── docs/                                 # algorithm and limitation notes
├── tests/                                # synthetic, data-free tests
└── third_party/                          # dependency and license notes
```

The repository intentionally does not include training scripts, experiment
logs, model-comparison configurations, alternative detector families, or the
multilabel classification component.

## Recommended environment

The public patient-level and analysis modules require:

- Ubuntu 22.04 LTS or WSL2
- Python 3.10 or newer
- NumPy 1.24 or newer

For users connecting the published configuration to MMDetection inference, the
reference software stack is:

| Component | Recommended version |
| --- | --- |
| Python | 3.10.x |
| PyTorch | 2.11.0 with CUDA 13.0 runtime |
| TorchVision | 0.26.0 with CUDA 13.0 runtime |
| MMDetection | 3.3.0 |
| MMCV | 2.1.0 with compiled CUDA operators |
| MMEngine | 0.10.7 |
| GPU | NVIDIA GPU with at least 16 GB VRAM; 24 GB or more preferred |

The package does not redistribute PyTorch, MMDetection, MMCV, pretrained
weights, or CUDA. Install those components according to their own official
documentation and licenses when detector inference is required.

## Installation

Create an isolated Python environment and install the public package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The core API depends only on NumPy. To use the Faster R-CNN configuration with
MMDetection, install the compatible PyTorch, TorchVision, MMEngine, MMCV and
MMDetection versions listed above separately.

## Usage

### Image-level detector output

The public interface accepts an MMDetection-style JSON record. `label` is the
zero-based detector class index and `box` uses `[x1, y1, x2, y2]` image
coordinates.

Example input:

```json
{
  "image_id": "radiograph-001",
  "detections": {
    "bboxes": [[120.0, 80.0, 310.0, 270.0]],
    "scores": [0.91],
    "labels": [0]
  }
}
```

Normalize and validate the record with:

```bash
python -m interface.infer_image input.json image_output.json
```

The output contains the class name, confidence, and validated image-space box.

### Patient-level four-layer analysis

Patient input groups all available radiographs for one patient and supplies the
detector detections. The optional `target_labels_by_image` field provides
patient-level reference labels. The optional `ground_truth_by_class` field
provides reference boxes when localization and post-localization typing
summaries are required.

```json
{
  "patient_id": "patient-001",
  "image_ids": ["radiograph-001", "radiograph-002"],
  "target_labels_by_image": {
    "radiograph-001": [0],
    "radiograph-002": []
  },
  "ground_truth_by_class": {
    "0": [{"box": [118.0, 79.0, 312.0, 272.0}]}
  },
  "detections": [
    {
      "image_id": "radiograph-001",
      "label": 0,
      "score": 0.91,
      "box": [120.0, 80.0, 310.0, 270.0]
    }
  ]
}
```

Run the patient-level interface with four fixed class thresholds:

```bash
python -m interface.infer_patient \
  patient_input.json patient_output.json \
  --threshold 0.50 0.50 0.50 0.50
```

The output includes patient-level class scores, fixed-threshold decisions, and
the four-layer result structure. Thresholds are part of the declared operating
point and should not be changed silently when comparing outputs.

### Unified five-state analysis

The existing `five_lesion_states.py` module and `summarize_patient_diagnosis`
interface handle both patient site summaries and reference-lesion outcomes.
For annotated images, run:

```bash
python -m interface.summarize_patient_diagnosis \
  examples/five_lesion_states_input.json lesion_analysis.json \
  --threshold 0.63 0.76 0.79 0.68
```

Each reference lesion receives exactly one outcome: no effective candidate,
localization insufficient, incorrect class, correct below threshold, or correct
with threshold met. Candidates have score >=0.05, with at most 100 retained
per image. Class-agnostic, score-ordered one-to-one localization requires
IoU >=0.50; an unmatched lesion is a localization failure if its best candidate
IoU is >=0.25, otherwise it has no effective candidate. Typing and frozen score
thresholds are checked on the matched prediction only.

The output declares the method and thresholds and provides overall/per-scene
counts, proportions and per-lesion matching records. The full definition is in
[`docs/FIVE_LESION_STATES.md`](docs/FIVE_LESION_STATES.md).

The patient interface also uses this same module. The complete synthetic
example exercises aggregation, all four layers and all five lesion outcomes:

```bash
python -m interface.infer_patient \
  examples/patient_input.json patient_output.json \
  --threshold 0.63 0.76 0.79 0.68
```

Its output contains `patient_site_summary`, the original `lesion_state` field,
and `five_lesion_analysis`. Layers 2, 3 and 4 use the same per-image lesion
matches. For multiple images, supply references inside each `images` entry or
through `ground_truth_by_image`; empty arrays explicitly declare negative
images. Class-grouped references without image IDs are supported for a single
image only.

### Patient-level site summary

Assign the mutually exclusive reporting state to an existing patient output:

```bash
python -m interface.summarize_patient_diagnosis \
  patient_output.json patient_diagnosis.json \
  --threshold 0.50 0.50 0.50 0.50
```

The patient-level site states are `no_core_fracture` and the four fracture-site
states. This original reporting function shares the module with lesion error
analysis, while retaining its patient unit and priority-based meaning.
The priority order is defined in
[`algorithm/schemas/five_lesion_states.yaml`](algorithm/schemas/five_lesion_states.yaml).

## Evaluation definitions

The analysis modules operate on validated detector outputs and reference boxes:

- [`category_specific_froc.py`](algorithm/diagnostic_analysis/category_specific_froc.py)
  uses score-ordered one-to-one matching within each fracture category.
- [`class_agnostic_froc.py`](algorithm/diagnostic_analysis/class_agnostic_froc.py)
  removes fracture-site identity from the matching step to expose type-related
  performance loss.
- [`iou_sensitivity.py`](algorithm/diagnostic_analysis/iou_sensitivity.py)
  evaluates the effect of the IoU acceptance criterion.
- [`localization_confusion_matrix.py`](algorithm/diagnostic_analysis/localization_confusion_matrix.py)
  separates correct typing, cross-site confusion, background detections, and
  missed lesions.
- [`five_lesion_states.py`](algorithm/diagnostic_analysis/five_lesion_states.py)
  provides patient site assignment, reference-lesion outcome assignment, and
  the unified summary entry used by both patient interfaces.

The formal data contracts are documented in
[`docs/INPUT_OUTPUT_SCHEMA.md`](docs/INPUT_OUTPUT_SCHEMA.md), while the
algorithmic definitions are in
[`docs/ALGORITHM.md`](docs/ALGORITHM.md) and
[`docs/FOUR_LAYER_DIAGNOSTIC_FRAMEWORK.md`](docs/FOUR_LAYER_DIAGNOSTIC_FRAMEWORK.md).

## Verification

The repository includes data-free unit tests for detector-output validation,
patient aggregation, all four diagnostic layers, FROC matching, IoU sensitivity,
localization confusion, original patient site assignment, and five lesion
outcomes. Tests cover score/IoU boundaries, competing lesions, stable score
ties, per-image isolation, and the integrated patient pipeline:

```bash
python -m unittest discover -s tests -v
```

These tests verify the algorithm contracts only. They do not reproduce article
performance without the governed clinical datasets and the corresponding model
checkpoint.

## Data and clinical boundaries

The clinical radiographs and patient-level metadata used for the article are
not distributed in this repository. Users must obtain lawful access to their
own data, remove direct and embedded identifiers, define patient-level grouping
before analysis, and comply with local ethics and data-governance requirements.

The model card and limitations are available in
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Citation

If you use this code, please cite the associated article:

```text
Deep Learning-based Detection of Ankle Fractures across Clinical Distribution
Shifts: A Multicenter Hierarchical Performance Evaluation.
```

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

The source code is released under the MIT License. Third-party frameworks,
pretrained weights, and clinical data remain subject to their own licenses and
access restrictions.
