from unittest import TestCase
import main as mn


class TestEvaluate(TestCase):
    def test_evaluate(self):
        list = ['aber das nächste mal doch bitte im querformat']
        result = mn.evaluate(list)
        self.assertEqual(result, 0)


