from unittest import TestCase
import matplotlib.pyplot as plt

from RandomBooleanCircuits import TruthTableAnalyser


class TestTruthTableAnalyser(TestCase):

    def test_fc_in_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_comp_count()
        res = analyser.fc_in_fp(inputs=4, outputs='[4, 5, 6, 7]', gates="['AND', 'OR', 'OR', 'AND']")
        self.assertTrue(res.item(), "Incorrect for circuit_description")
        res = analyser.fc_in_fp()
        self.assertFalse(res.values[0], "Incorrect for full frame")

    def test_find_fc(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_comp_count()
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'OR', 'AND']")]
        self.assertTrue(analyser.find_fc(circuit))
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'AND', 'AND']")]
        self.assertFalse(analyser.find_fc(circuit))

    def test_find_fc_tt(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_comp_count()
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'OR', 'AND']")]
        self.assertTrue(analyser.find_fc_tt(circuit))
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'AND', 'AND']")]
        self.assertTrue(analyser.find_fc_tt(circuit))

    def test_fc_tt_in_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        res = analyser.fc_tt_in_fp(inputs=4, outputs='[4, 5, 6, 7]', gates="['AND', 'OR', 'OR', 'AND']")
        self.assertTrue(res.item(), "Incorrect for circuit_description with fc in fp")
        res = analyser.fc_tt_in_fp(inputs=4, outputs='[4, 5, 6, 7]', gates="['AND', 'OR', 'AND', 'AND']")
        self.assertTrue(res.item(), "Incorrect for circuit_description with fc not in fp")
        res = analyser.fc_tt_in_fp()
        self.assertTrue(res.values[0], "Incorrect for full frame")

    def test_av_dist_fc_to_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        res = analyser.av_dist_fc_to_fp()
        self.assertEqual(res.values[1], 2.0, "Average distance from full circuit_description is wrong")

    def test_av_dist_btw_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        res = analyser.av_dist_btw_fp()
        self.assertEqual(res.values[2], 2.0, "Average distance between final population is wrong")

    def test_max_dist_btw_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        res = analyser.max_dist_btw_fp()
        self.assertEqual(res.values[2], 4.0, "Max distance between final population is wrong")
        self.assertEqual(res.values[0], 3.0, "Max distance between final population is wrong")

    def test_av_tt_dist_btw_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        res = analyser.av_tt_dist_btw_fp()
        self.assertEqual(res.values[0], 0.0, "Incorrect average truth table dist for final populations")

    def test_max_tt_dist_btw_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        res = analyser.max_tt_dist_btw_fp()
        self.assertEqual(res.values[0], 0, "Max truth table distance in final population")
        self.assertEqual(res.values[2], 16, "Max truth table distance in final pop incorrect")

    def test_av_tt_dist_to_fc(self):
        analyser = TruthTableAnalyser("TestData.csv")
        res = analyser.av_tt_dist_fc_to_fp()
        self.assertEqual(res.values[1], 0.5, "Average tt distance from full circuit_description is wrong")

    def test_max_comp_in_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_comp_count()
        res = analyser.max_comp_in_fp()
        self.assertEqual(4, res.values[2], "Incorrect max components in circuit_description")

    def test_min_comp_in_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_comp_count()
        res = analyser.min_comp_in_fp()
        self.assertEqual(0, res.values[2], "Incorrect min components in circuit_description")

    def test_mode_comp_in_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_comp_count()
        res = analyser.mode_comp_in_fp()
        self.assertEqual(2, res.values[2], "Incorrect mode components in circuit_description")

    def test_median_comp_in_fp(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_comp_count()
        res = analyser.median_comp_in_fp()
        self.assertEqual(2, res.values[2], "Incorrect median components in circuit_description")

    def test_add_comp_count(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_comp_count()
        self.assertIn("Comp_Count", analyser.df.columns, "Failed to add component count")
        self.assertEqual(2, analyser.df.Comp_Count[0], "Incorrect component count")

    def test_add_comp_fraction(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_comp_fraction()
        self.assertIn("Comp_Frac", analyser.df.columns, "Failed to add component ratio")
        self.assertEqual(0.5, analyser.df.Comp_Frac[0], "Incorrect ratio")
        self.assertEqual(0.75, analyser.df.Comp_Frac[29], "Incorrect ratio")

    def test_add_fc_dist(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_fc_dist()
        self.assertIn("FC_Dist", analyser.df.columns, "Failed to add parameter distance from full circuit_description")
        self.assertEqual(2, analyser.df.FC_Dist[0], "Incorrect count of distance")
        self.assertEqual(3, analyser.df.FC_Dist[20], "Incorrect count of distance")

    def test_add_fc_tt_dist(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.add_fc_tt_dist()
        self.assertEqual(16, analyser.df["FC_TT_Dist"][0], "Incorrect full circuit_description test table distance")

    def test_add_fc_tt_dist_2(self):
        analyser = TruthTableAnalyser("4_gate_test_data.csv")
        analyser.add_fc_tt_dist()
        self.assertIn("FC_TT_Dist", analyser.df.columns, "Column not added for 4 gate output")

    def test_calc_dist_2(self):
        analyser = TruthTableAnalyser("TestData.csv")
        self.assertEqual(1, analyser.calc_dist_2(0, 1), "Incorrect distance, simple")
        self.assertEqual(4, analyser.calc_dist_2(11, 12), "Incorrect distance, harder")

    def test_calc_dist_group(self):
        analyser = TruthTableAnalyser("TestData.csv")
        self.assertEqual(2.0, analyser.calc_dist_group(inputs=4, outputs='[4, 5, 6, 7]',
                                                       gates="['AND', 'OR', 'OR', 'AND']", final=True),
                         "Single circuit_description final pop average distance wrong")
        self.assertAlmostEqual(2.13333, analyser.calc_dist_group(inputs=4, outputs='[4, 5, 6, 7]',
                                                                 gates="['AND', 'OR', 'OR', 'AND']"), 5,
                               "Single circuit_description complete population average distance wrong")
        self.assertAlmostEqual(2.042553, analyser.calc_dist_group(), 5, "Complete results average distance wrong")

    def test_calc_dist_circuit(self):
        analyser = TruthTableAnalyser("TestData.csv")
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'OR', 'AND']") &
                                  (analyser.df["Final_Population"])]
        self.assertEqual(2.0, analyser.calc_dist_circuit(circuit), "Incorrect for single circuit_description final pop")
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'OR', 'AND']")]
        self.assertAlmostEqual(2.13333, analyser.calc_dist_circuit(circuit), 5,
                               "Single circuit_description complete population average distance wrong")
        self.assertAlmostEqual(2.042553, analyser.calc_dist_circuit(analyser.df), 5,
                               "Complete results average distance wrong")

    def test_calc_max_dist_circuit(self):
        analyser = TruthTableAnalyser("TestData.csv")
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'OR', 'AND']") &
                                  (analyser.df["Final_Population"])]
        self.assertEqual(4.0, analyser.calc_max_dist_circuit(circuit), "Incorrect for single circuit_description final pop")
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'OR', 'AND']")]
        self.assertEqual(4.0, analyser.calc_max_dist_circuit(circuit),
                         "Single circuit_description complete population average distance wrong")
        self.assertEqual(4.0, analyser.calc_max_dist_circuit(analyser.df),
                         "Complete results average distance wrong")

    def test_calc_max_tt_dist_circuit(self):
        analyser = TruthTableAnalyser("TestData.csv")
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'OR', 'AND']") &
                                  (analyser.df["Final_Population"])]
        self.assertEqual(16.0, analyser.calc_max_tt_dist_circuit(circuit),
                         "Incorrect tt distance for single circuit_description final pop")
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'OR', 'AND']")]
        self.assertEqual(16.0, analyser.calc_max_tt_dist_circuit(circuit),
                         "Single circuit_description complete population average tt distance wrong")
        self.assertEqual(16.0, analyser.calc_max_tt_dist_circuit(analyser.df),
                         "Complete results average tt distance wrong")

    def test_calc_tt_dist_2(self):
        analyser = TruthTableAnalyser("TestData.csv")
        self.assertEqual(12, analyser.calc_tt_dist_2(0, 1), "Incorrect distance, simple")
        self.assertEqual(16, analyser.calc_tt_dist_2(11, 12), "Incorrect distance, harder")

    def test_truth_to_list(self):
        analyser = TruthTableAnalyser("TestData.csv")
        self.assertEqual(0, analyser.df.at[0, "Truth_Table"][0])

    def test_calc_tt_dist_circuit(self):
        analyser = TruthTableAnalyser("TestData.csv")
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'AND', 'AND']") &
                                  (analyser.df["Final_Population"])]
        self.assertEqual(0.0, analyser.calc_tt_dist_circuit(circuit), "Incorrect for single circuit_description final pop")
        circuit = analyser.df.loc[(analyser.df["Input_Count"] == 4) & (analyser.df["Outputs"] == '[4, 5, 6, 7]') &
                                  (analyser.df["Gates"] == "['AND', 'OR', 'AND', 'AND']")]
        self.assertAlmostEqual(2.508333, analyser.calc_tt_dist_circuit(circuit), 5,
                               "Single circuit_description complete population average distance wrong")
        self.assertAlmostEqual(6.98797409, analyser.calc_tt_dist_circuit(analyser.df), 5,
                               "Complete results average distance wrong")

    def test_gates_to_label(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.gates_to_label()
        self.assertIn("Labels", analyser.df.columns, "Labels not in columns")
        self.assertEqual(6, analyser.df.at[0, "Labels"], "Incorrect label assigned")

    def test_tt_distance_graph(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.tt_distance_graph()
        plt.show()

    def test_comp_distance_graph(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.comp_distance_graph()
        plt.show()

    def test_circuit_graph(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.circuit_graph()
        plt.show()

    def test_variants_by_fc(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.variants_by_fc()
        plt.show()

    def test_comp_size_by_tt(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.comp_size_by_tt()
        plt.show()

    def test_three_point_circuit_dist(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.three_point_circuit_dist()
        plt.show()

    def test_tt_by_circuit(self):
        analyser = TruthTableAnalyser("TestData.csv")
        analyser.tt_by_circuit()
        plt.show()
