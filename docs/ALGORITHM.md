# Algorithm definition

The detector is Faster R-CNN with a ResNet-50-FPN feature extractor and four
fracture-site outputs. Each detection contains an image identifier, class,
confidence and image-space box.

The public MMDetection test pipeline uses aspect-ratio-preserving resize with
scale (1333, 800). The model preprocessor uses RGB means
[123.675, 116.28, 103.53], standard deviations [58.395, 57.12, 57.375],
BGR-to-RGB conversion and right/bottom padding to a divisor of 32. Class-wise
NMS uses IoU 0.50, score filtering uses 0.05 and at most 100 boxes are retained
per image. The geometry helper uses actual rounded dimensions for inverse
coordinate mapping. Detector outputs supplied to analysis use original-image
coordinates.

Patient scores take the maximum confidence per fracture site across the
patient's radiographs. Declared class thresholds yield four binary patient
site decisions and the original priority-based patient reporting state.

The four layers are patient fracture recognition, same-class lesion detection,
post-localization fracture-site typing, and fixed-threshold detection.
Layer 2 uses per-image same-class matching and pooled FROC operating points.
With image-grouped reference annotations, layers 3–4 share the matches from
`five_lesion_states.py`: per-image, descending-score, class-agnostic one-to-one
matching at IoU >=0.50, using candidates with score >=0.05 and at most 100
predictions per image. Stable ties follow class order and then input order.

Every reference lesion receives one outcome: no effective candidate,
localization insufficient, wrong class, correct below the frozen threshold,
or correct with the threshold met. The effective-candidate gate is IoU >=0.25;
site identity and the class threshold are tested on the matched box. Default
thresholds are lateral 0.63, medial 0.76, posterior 0.79 and distal tibia 0.68.

The original patient site summary is preserved in this same analysis module.
It does not alter the detector classes and must not be interpreted as lesion
failure attribution. Category-specific FROC, class-agnostic FROC and IoU
sensitivity remain complementary detector analyses. Exact five-state rules,
inputs and outputs are in [FIVE_LESION_STATES.md](FIVE_LESION_STATES.md).

The standalone localization confusion matrix also uses the canonical
class-agnostic matcher, including its score floor, top-100 limit and tie order.
Each prediction selects the highest-IoU available reference. The standalone
interface accepts one image; patient-level analysis uses image-grouped inputs.

Threshold selection maximizes validation patient-level F1 over 0.05 through
0.95 inclusive in steps of 0.01. Ties favor the threshold closest to 0.50,
then the higher threshold. The four selected thresholds are frozen for testing.
