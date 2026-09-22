# Algorithm definition

The detector is Faster R-CNN with a ResNet-50-FPN feature extractor and four
fracture-site outputs. The detector emits a class, confidence, and image-space
box for each candidate lesion.

Patient-level scores are obtained by taking the maximum confidence for each
fracture site across all images belonging to the patient. A fixed class-wise
threshold converts those scores into patient-level recognition decisions.

The four diagnostic layers are:

1. **Patient fracture recognition**: determine whether each target fracture
   site is present for the patient.
2. **Lesion localization**: match image-level boxes to reference lesions with
   one-to-one, score-ordered matching at a declared IoU threshold.
3. **Post-localization typing**: compare the predicted fracture site with the
   reference site for matched and competing boxes.
4. **Fixed-threshold detection**: report the final operating point using the
   declared score thresholds.

Category-specific FROC, class-agnostic FROC, IoU sensitivity, and the
localization confusion matrix are diagnostic views of the same detector output.
The five lesion states are a mutually exclusive reporting taxonomy and are
not a fifth detector output.
