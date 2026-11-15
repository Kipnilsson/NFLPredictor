import unittest

from src.jev import client, prompt_study, template


class PromptStudyTests(unittest.TestCase):
    def test_variants_change_only_the_intended_request_fields(self):
        model, state = "jev-1.13.0", "A fixed pre-snap situation."
        baseline = client.make_payload(model, state, template.labels())
        numeric, frequency, labels = [
            prompt_study.make_payload(model, state, variant)
            for variant in prompt_study.VARIANTS
        ]
        self.assertEqual(baseline, numeric)
        self.assertEqual(len({client.payload_key(p) for p in (numeric, frequency, labels)}), 3)
        for payload in (frequency, labels):
            self.assertEqual(payload["state"], state)
            self.assertEqual(payload["model"], model)
            self.assertEqual(list(payload["questions"]["next_play"]["criteria"]), template.labels())

        original = numeric["questions"]["next_play"]
        frequency_question = frequency["questions"]["next_play"]
        label_question = labels["questions"]["next_play"]
        self.assertEqual(frequency_question["criteria"], original["criteria"])
        self.assertEqual(label_question["instructions"], original["instructions"])
        self.assertNotEqual(frequency_question["instructions"], original["instructions"])
        self.assertNotEqual(label_question["criteria"], original["criteria"])
