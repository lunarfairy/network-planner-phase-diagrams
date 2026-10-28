import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'llm_pilot'))


class CloudRecordsTests(unittest.TestCase):
    def test_historical_comparison_is_available_without_missing_panels(self):
        self.assertTrue((ROOT / 'code/archived_history.py').is_file(),
                        'The release must load the cloud historical comparison without fits_old.csv or panel_old.csv.gz.')
        load = importlib.import_module('archived_history').load_historical_comparison
        likelihoods, fits = load()
        self.assertEqual(set(likelihoods['what']), {'old_panel_old_M1_params'})
        self.assertEqual(likelihoods['Q'].tolist(), [24, 48, 96, 160, 240, 320, 480])
        self.assertAlmostEqual(likelihoods.loc[likelihoods.Q == 24, 'nll'].iloc[0], 6788.2475691991285)
        self.assertTrue(likelihoods.loc[likelihoods.Q == 480, 'nll'].isna().all())
        self.assertEqual(fits['subset'].tolist(), ['oldpanel_Q160', 'oldpanel_Q24'])
        self.assertEqual(len(fits), 2)
        self.assertFalse((ROOT / 'fits_old.csv').exists())
        self.assertFalse((ROOT / 'panel_old.csv.gz').exists())

    def test_saved_pilot_records_are_the_replay_input(self):
        self.assertTrue((ROOT / 'llm_pilot/replay_records.py').is_file(),
                        'The release needs an offline replay of the actual 280 cloud records.')
        summarize = importlib.import_module('replay_records').summarize
        records = json.loads((ROOT / 'llm_pilot/llm_elicitation_responses.json').read_text(encoding='utf-8'))
        result = summarize(records)
        self.assertEqual(result['records'], 280)
        self.assertEqual(result['tasks'], {'coop': 240, 'accept': 40})
        self.assertEqual(result['conditions'], 28)
        self.assertEqual(result['repetitions_per_condition'], [10])
        self.assertEqual(result['parsed'], 226)
        self.assertEqual(result['models'], ['claude-haiku-4-5-20251001'])


if __name__ == '__main__':
    unittest.main()
