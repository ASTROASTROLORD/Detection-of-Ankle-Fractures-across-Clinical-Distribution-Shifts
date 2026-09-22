import unittest

from algorithm.detector.detection_output import parse_detections


class DetectorOutputTests(unittest.TestCase):
    def test_valid_output_is_normalized(self):
        detections = parse_detections("image-1", {"bboxes": [[0, 0, 10, 10]], "scores": [0.8], "labels": [2]})
        self.assertEqual(detections[0].class_name, "posterior_malleolus_fracture")

    def test_invalid_label_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_detections("image-1", {"bboxes": [[0, 0, 10, 10]], "scores": [0.8], "labels": [4]})


if __name__ == "__main__":
    unittest.main()
