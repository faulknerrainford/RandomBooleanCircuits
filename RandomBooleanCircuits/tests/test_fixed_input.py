"""
:author: Penn Faulkner Rainford
:license: GPL v3
:copyright: 2022-2023
"""

from unittest import TestCase
from RandomBooleanCircuits import RBCSystem
from RandomBooleanCircuits import fixed_input as fin
import numpy as np


class Test(TestCase):
    def test_fixed_input(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 2, 5], [0, 4, 3, 6], [1, 5, 4, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 0, 0, 0])), "binary")
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 0, 0, 1])), "binary")
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 0, 1, 0])), "binary")
        self.assertEqual(4.0, fin.fixed_input_binary(system, np.array([0, 0, 1, 1])), "binary")
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 1, 0, 0])), "binary")
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 1, 0, 1])), "binary")
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 1, 1, 0])), "binary")
        self.assertEqual(5.0, fin.fixed_input_binary(system, np.array([0, 1, 1, 1])), "binary")
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([1, 0, 0, 0])), "binary")
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([1, 0, 0, 1])), "binary")
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([1, 0, 1, 0])), "binary")
        self.assertEqual(4.0, fin.fixed_input_binary(system, np.array([1, 0, 1, 1])), "binary")
        self.assertEqual(10.0, fin.fixed_input_binary(system, np.array([1, 1, 0, 0])), "binary")
        self.assertEqual(10.0, fin.fixed_input_binary(system, np.array([1, 1, 0, 1])), "binary")
        self.assertEqual(10.0, fin.fixed_input_binary(system, np.array([1, 1, 1, 0])), "binary")
        self.assertEqual(15.0, fin.fixed_input_binary(system, np.array([1, 1, 1, 1])), "binary")

    def test_fixed_input_binary(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 2, 5], [0, 5, 3, 6], [1, 6, 4, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 0, 0, 0])))
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 0, 0, 1])))
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 0, 1, 0])))
        self.assertEqual(4.0, fin.fixed_input_binary(system, np.array([0, 0, 1, 1])))
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 1, 0, 0])))
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 1, 0, 1])))
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([0, 1, 1, 0])))
        self.assertEqual(4.0, fin.fixed_input_binary(system, np.array([0, 1, 1, 1])))
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([1, 0, 0, 0])))
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([1, 0, 0, 1])))
        self.assertEqual(0.0, fin.fixed_input_binary(system, np.array([1, 0, 1, 0])))
        self.assertEqual(6.0, fin.fixed_input_binary(system, np.array([1, 0, 1, 1])))
        self.assertEqual(8.0, fin.fixed_input_binary(system, np.array([1, 1, 0, 0])))
        self.assertEqual(8.0, fin.fixed_input_binary(system, np.array([1, 1, 0, 1])))
        self.assertEqual(8.0, fin.fixed_input_binary(system, np.array([1, 1, 1, 0])))
        self.assertEqual(15.0, fin.fixed_input_binary(system, np.array([1, 1, 1, 1])))

    def test_fixed_input_binary_0(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary_0(system))

    def test_fixed_input_binary_1(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary_1(system))

    def test_fixed_input_binary_2(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary_2(system))

    def test_fixed_input_binary_3(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(4.0, fin.fixed_input_binary_3(system))

    def test_fixed_input_binary_4(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary_4(system))

    def test_fixed_input_binary_5(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary_5(system))

    def test_fixed_input_binary_6(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary_6(system))

    def test_fixed_input_binary_7(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(5.0, fin.fixed_input_binary_7(system))

    def test_fixed_input_binary_8(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary_8(system))

    def test_fixed_input_binary_9(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary_9(system))

    def test_fixed_input_binary_10(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(0.0, fin.fixed_input_binary_10(system))

    def test_fixed_input_binary_11(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(4.0, fin.fixed_input_binary_11(system))

    def test_fixed_input_binary_12(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(10.0, fin.fixed_input_binary_12(system))

    def test_fixed_input_binary_13(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(10.0, fin.fixed_input_binary_13(system))

    def test_fixed_input_binary_14(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(10.0, fin.fixed_input_binary_14(system))

    def test_fixed_input_binary_15(self):
        system = RBCSystem(input_units=1, output_units=1, hidden_units=1, wash_out=1,
                           circuitSettings=[4, [[0, 1, 1, 4], [2, 3, 3, 5], [0, 4, 5, 6], [1, 5, 6, 7]], [4, 5, 6, 7]])
        self.assertEqual(15.0, fin.fixed_input_binary_15(system))
