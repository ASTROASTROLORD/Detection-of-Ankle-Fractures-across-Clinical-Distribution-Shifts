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
