"""
:author: Penn Faulkner Rainford
:license: GPL v3
:copyright: 2022-2023
"""

from unittest import TestCase
from RandomBooleanCircuits import Circuit


class TestCircuit(TestCase):
    def test_run_gates(self):
        and_circuit = Circuit("0 1, 0 1 1 2, 2")
        self.assertEqual([0], and_circuit.run("00"), "Failed AND 00")
        self.assertEqual([0], and_circuit.run("01"), "Failed AND 01")
        self.assertEqual([0], and_circuit.run("10"), "Failed AND 10")
        self.assertEqual([1], and_circuit.run("11"), "Failed AND 11")
        nand_circuit = Circuit("0 1, 0 1 2 2, 2")
        self.assertEqual([1], nand_circuit.run("00"), "Failed NAND 00")
        self.assertEqual([1], nand_circuit.run("01"), "Failed NAND 01")
        self.assertEqual([1], nand_circuit.run("10"), "Failed NAND 10")
        self.assertEqual([0], nand_circuit.run("11"), "Failed NAND 11")
        or_circuit = Circuit("0 1, 0 1 3 2, 2")
        self.assertEqual([0], or_circuit.run("00"), "Failed OR 00")
        self.assertEqual([1], or_circuit.run("01"), "Failed OR 01")
        self.assertEqual([1], or_circuit.run("10"), "Failed OR 10")
        self.assertEqual([1], or_circuit.run("11"), "Failed OR 11")
        nor_circuit = Circuit("0 1, 0 1 4 2, 2")
        self.assertEqual([1], nor_circuit.run("00"), "Failed NOR 00")
        self.assertEqual([0], nor_circuit.run("01"), "Failed NOR 01")
        self.assertEqual([0], nor_circuit.run("10"), "Failed NOR 10")
        self.assertEqual([0], nor_circuit.run("11"), "Failed NOR 11")
        xor_circuit = Circuit("0 1, 0 1 5 2, 2")
        self.assertEqual([0], xor_circuit.run("00"), "Failed XOR 00")
        self.assertEqual([1], xor_circuit.run("01"), "Failed XOR 01")
        self.assertEqual([1], xor_circuit.run("10"), "Failed XOR 10")
        self.assertEqual([0], xor_circuit.run("11"), "Failed XOR 11")
        nxor_circuit = Circuit("0 1, 0 1 6 2, 2")
        self.assertEqual([1], nxor_circuit.run("00"), "Failed NXOR 00")
        self.assertEqual([0], nxor_circuit.run("01"), "Failed NXOR 01")
        self.assertEqual([0], nxor_circuit.run("10"), "Failed NXOR 10")
        self.assertEqual([1], nxor_circuit.run("11"), "Failed NXOR 11")

    def test_run_circuits(self):
        circuit1 = Circuit("0 1 2, 0 1 1 3, 2 3 3 4, 4")
        self.assertEqual([0], circuit1.run("000"), "Failed 000 single output")
        self.assertEqual([0], circuit1.run("010"), "Failed 010 single output")
        self.assertEqual([0], circuit1.run("100"), "Failed 100 single output")
        self.assertEqual([1], circuit1.run("110"), "Failed 110 single output")
        self.assertEqual([1], circuit1.run("001"), "Failed 001 single output")
        self.assertEqual([1], circuit1.run("011"), "Failed 011 single output")
        self.assertEqual([1], circuit1.run("101"), "Failed 101 single output")
        self.assertEqual([1], circuit1.run("111"), "Failed 111 single output")
        circuit2 = Circuit("0 1 2, 0 1 1 3, 2 3 3 4, 3 4")
        self.assertEqual([0, 0], circuit2.run("000"), "Failed 000 dual output")
        self.assertEqual([0, 0], circuit2.run("010"), "Failed 010 dual output")
        self.assertEqual([0, 0], circuit2.run("100"), "Failed 100 dual output")
        self.assertEqual([1, 1], circuit2.run("110"), "Failed 110 dual output")
        self.assertEqual([0, 1], circuit2.run("001"), "Failed 001 dual output")
        self.assertEqual([0, 1], circuit2.run("011"), "Failed 011 dual output")
        self.assertEqual([0, 1], circuit2.run("101"), "Failed 101 dual output")
        self.assertEqual([1, 1], circuit2.run("111"), "Failed 111 dual output")
