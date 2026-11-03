# Unreleased — manuscript alignment

The project display title and citation title now match the manuscript
“Multicenter Evaluation of Patient-Level Recognition and Lesion-Level Detection
of Ankle Fractures across Clinical Scenarios.” Citation metadata points to the
public repository.

Level 2 now uses per-image same-class matching at IoU >=0.50 and provides pooled
FROC operating points. Layers 3 and 4 use class-agnostic matching for conditional
typing and five-outcome analysis. Existing Level 2 output keys are retained;
their localization counts and sensitivity now require correct site class.
The new detection fields express this definition explicitly. Cohort evaluation
includes negative images in the FP/radiograph denominator.

The default validation threshold grid is 0.05–0.95 inclusive at 0.01 increments.
README patient examples use the manuscript's frozen class thresholds and
image-grouped references. Regression tests cover these method definitions.

The standalone localization confusion matrix now shares the canonical matcher
with the patient pipeline. It searches available references after a competing
prediction occupies the highest-IoU reference, applies the score floor and
top-100 limit, and rejects mixed-image inputs. Custom IoU thresholds remain
supported.

The geometry helper now defaults to the manuscript's 800/1333 resize bounds,
supports upscaling, and accounts for rounded per-axis scales when mapping boxes
back to the source. Padding is right/bottom to multiples of 32. The public
MMDetection configuration includes the complete inference transform pipeline,
explicit normalization and class-wise NMS settings.

# Version 0.2.0

The original five-state module, summary interface, documentation and tests now
include hierarchical reference-lesion outcome analysis. The original patient
fracture-site summary remains available through its public API and output field.

The patient pipeline uses the same lesion matching result for localization,
conditional site typing and five-outcome threshold analysis. Multi-image
references retain image identity; ungrouped class references are accepted only
for one image. Patient input now supports both its documented images form and
the original image_ids plus flat detections form. Patient output includes the
class_predictions and layers fields declared by its schema.

The release includes synthetic inputs, complete input/output schemas, fixed
threshold metadata and boundary/competition/image-isolation/integration tests.
The installed package includes the interface modules and schema resources.

The five-lesion YAML now groups reference-lesion outcomes and patient site
states explicitly. Users reading its previous top-level states/priority_order
keys should use patient_site_summary.states/priority_order. Function calls and
existing patient summary commands remain supported.
