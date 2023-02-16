"""
:author: Penn Faulkner Rainford
:license: GPL v3
:copyright: 2022-2023
"""

from unittest import TestCase

from RandomBooleanCircuits import circuit_search, add_search_results
from pycharc.search.microbial_ga import MicrobialGA

import pandas as pd


class Test(TestCase):
    def test_circuit_search(self):
        search = circuit_search(["AND", "OR", "OR", "AND"])
        self.assertIsInstance(search, MicrobialGA, "Incorrect search type returned.")

    def test_add_search_results(self):
        df = pd.DataFrame(columns=["Input_Count", "Outputs", "Wiring", "Gates", "Gate_Activations", "Truth_Table",
                                   "Final_Population"])
        df = df.astype({"Final_Population": bool})
        search = circuit_search(["AND", "OR", "OR", "AND"])
        search.run()
        df = add_search_results(["AND", "OR", "OR", "AND"], search, df)
        self.assertIn("Outputs", df.columns, "Incorrect Columns")
        self.assertGreaterEqual(df.shape[0], 12, "Incorrect Number of Rows")
