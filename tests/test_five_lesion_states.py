import unittest

from algorithm.diagnostic_analysis.five_lesion_states import assign_lesion_state


class FiveLesionStateTests(unittest.TestCase):
    def test_priority_produces_one_state(self):
        scores = {"lateral_malleolus_fracture": 0.9, "distal_tibial_fracture": 0.8}
        thresholds = {name: 0.5 for name in scores}
        self.assertEqual(assign_lesion_state(scores, thresholds), "distal_tibial_fracture")

    def test_no_positive_is_normal_state(self):
        self.assertEqual(assign_lesion_state({}, {}), "no_core_fracture")


if __name__ == "__main__":
    unittest.main()
