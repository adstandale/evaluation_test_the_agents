import unittest

from deep_eval import evaluate_agent_communication, evaluate_agent_response


class DeepEvalIntegrationTests(unittest.TestCase):
    def test_evaluate_agent_communication_without_deepeval_skips_cleanly(self):
        result = evaluate_agent_communication(
            {
                "sender": "main_agent",
                "receiver": "crew_agent",
                "type": "travel_request",
                "payload": {"country_name": "India", "user_query": "Trip to Goa"},
            },
            expected_summary="A travel request containing country and trip details.",
        )

        self.assertIn(result["status"], {"skipped", "ok"})

    def test_evaluate_agent_response_for_final_output(self):
        result = evaluate_agent_response(
            agent_name="final_agent",
            user_input="Trip to Goa for 3 days",
            response="Day 1: fly to Goa; Day 2: beach and hotel stay; Day 3: return flight.",
            expected_output="A short Goa itinerary with flights, hotel stay, and departure details.",
        )

        self.assertIn(result["status"], {"skipped", "ok"})


if __name__ == "__main__":
    unittest.main()
