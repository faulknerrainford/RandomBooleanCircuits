from unittest import TestCase
from RandomBooleanCircuits import CircuitExplorer


class TestCircuitExplorer(TestCase):
    def test_run(self):
        circuit_X = CircuitExplorer(3, [[0, 1, 1, 3], [2, 3, 3, 4]], [4])
        self.assertEqual([[0], [0], [0], [0], [0], [0], [0], [1]], circuit_X.run("11"), "Failed full circuit_description")
        self.assertEqual([[0], [1], [0], [1], [0], [1], [0], [1]], circuit_X.run("10"), "Failed AND circuit_description")
        self.assertEqual([[0], [0], [0], [0], [0], [1], [0], [1]], circuit_X.run("01"), "Failed OR circuit_description")
        self.assertEqual([[0], [1], [0], [1], [0], [1], [0], [1]], circuit_X.run("00"), "Failed EMPTY circuit_description")
        circuit_X = CircuitExplorer(3, [[0, 1, 1, 3], [2, 3, 3, 4]], [3, 4])
        self.assertEqual([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 1]], circuit_X.run("11"),
                         "Failed full circuit_description 2 output")
        self.assertEqual([[0, 0], [0, 1], [0, 0], [0, 1], [0, 0], [0, 1], [1, 0], [1, 1]], circuit_X.run("10"),
                         "Failed AND circuit_description 2 output")
        self.assertEqual([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 1], [1, 0], [1, 1]], circuit_X.run("01"),
                         "Failed OR circuit_description 2 output")
        self.assertEqual([[0, 0], [0, 1], [0, 0], [0, 1], [1, 0], [1, 1], [1, 0], [1, 1]], circuit_X.run("00"),
                         "Failed EMPTY circuit_description 2 output")

    def test_gen_binary_strings(self):
        circuit_X = CircuitExplorer(3, [[0, 1, 1, 3], [2, 3, 3, 4]], [4])
        self.assertEqual(["000", "001", "010", "011", "100", "101", "110", "111"], circuit_X.gen_binary_strings(),
                         "Incorrect three bit string generation")
        circuit_X = CircuitExplorer(5, [[0, 1, 1, 5], [2, 3, 3, 6]], [5])
        self.assertEqual(["00000", "00001", "00010", "00011", "00100", "00101", "00110", "00111", "01000", "01001",
                          "01010", "01011", "01100", "01101", "01110", "01111", "10000", "10001", "10010", "10011",
                          "10100", "10101", "10110", "10111", "11000", "11001", "11010", "11011", "11100", "11101",
                          "11110", "11111"], circuit_X.gen_binary_strings(), "Incorrect five bit string generation")
