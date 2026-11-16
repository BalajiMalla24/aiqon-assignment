import unittest
from datetime import datetime

from eta_model import fit_and_evaluate, generate_shipments, predict_shipment


class ShipmentEtaModelTests(unittest.TestCase):
    def setUp(self):
        self.shipments = generate_shipments(80, seed=7)

    def test_generation_is_reproducible(self):
        self.assertEqual(self.shipments, generate_shipments(80, seed=7))

    def test_recorded_final_delay_matches_timestamp_difference(self):
        shipment = self.shipments[0]
        scheduled = datetime.fromisoformat(shipment["scheduled_arrival"])
        actual = datetime.fromisoformat(shipment["actual_arrival"])
        observed_delay = (actual - scheduled).total_seconds() / 3600
        self.assertAlmostEqual(observed_delay, float(shipment["final_delay_hours"]), delta=0.02)

    def test_model_evaluates_and_returns_explained_prediction(self):
        models, metrics = fit_and_evaluate(self.shipments)
        prediction = predict_shipment(models, self.shipments[0])

        self.assertEqual(metrics["test_shipments"], 20)
        self.assertGreater(metrics["mae_hours"], 0)
        self.assertGreaterEqual(prediction["late_probability"], 0)
        self.assertLessEqual(prediction["late_probability"], 1)
        self.assertEqual(
            [abs(item["hours"]) for item in prediction["contributions"]],
            sorted((abs(item["hours"]) for item in prediction["contributions"]), reverse=True),
        )
        self.assertIsInstance(prediction["predicted_eta"], datetime)


if __name__ == "__main__":
    unittest.main()