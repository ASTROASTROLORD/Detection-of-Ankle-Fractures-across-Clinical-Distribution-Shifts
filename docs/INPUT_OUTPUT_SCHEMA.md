# Input and output schema

The public interfaces use JSON-compatible records. A patient input contains a
stable patient identifier, image identifiers, optional reference labels, and
detector detections. Each detection contains `image_id`, integer `label`, a
confidence `score` in `[0, 1]`, and an `[x1, y1, x2, y2]` box.

The output contains one score and one fixed-threshold decision for each of the
four fracture sites, the four layer summaries, and the optional five-state
diagnostic label. Schemas are stored in `algorithm/schemas/`.
