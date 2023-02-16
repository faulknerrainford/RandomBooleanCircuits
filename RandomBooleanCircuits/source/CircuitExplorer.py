"""
:author: Penn Faulkner Rainford
:license: GPL v3
:copyright: 2022-2023
"""

from RandomBooleanCircuits import Circuit


class CircuitExplorer:
    """
    CircuitExplorer can test a given input or the full range of possible inputs to a circuit_description.

    Parameters
    ----------
    inputs  :   int
        The number of inputs into the circuit_description
    gates   :   list<list<int, int, int, int>>
        The list of the gates given as: [inputA, inputB, gateType, output]
    outputs :   list<ints>
        The list of outputs from the circuit_description by integer label
    """

    def __init__(self, inputs, gates, outputs):
        # set up circuit_description
        self.gates = gates
        self.input_count = inputs
        circuit_input = ""
        for i in range(inputs):
            circuit_input = circuit_input + " " + str(i)
        self.circuit_input = circuit_input[1:] + ", "
        circuit_output = ""
        pass
        for o in outputs:
            if o == outputs[0]:
                circuit_output = circuit_output + str(o)
            else:
                circuit_output = circuit_output + " " + str(o)
        self.circuit_output = circuit_output
        self.input_values = self.gen_binary_strings()
        self.parameters = None

    def run(self, parameters):
        """
        Takes a set of parameters for gate activity (inactive gates buffer first input) and runs all possible inputs to
        produce truth table.

        Parameters
        ----------
        parameters  :   list<0/1>
            List of binary values with length equal to gates. Indicates activation of gates.

        Returns
        -------
        list<list<0/1>>
            List of binary results of outputs from each possible input.
        """
        # turn gates off by replacing gates with zero based on parameters
        circuit = Circuit(self.circuit_input + self.set_parameters(parameters) + self.circuit_output)
        # run all inputs on circuit_description and return an ordered truth table
        truth_table = []
        for bit_string in self.input_values:
            truth_table.append(circuit.run(bit_string))
        return truth_table

    def run_single(self, inputs, parameters=None):
        """
        Takes a set of parameters for gate activity (inactive gates buffer first input) and run the given input on the
        resultant circuit_description and returns the list of outputs.

        Parameters
        -----------
        inputs  :   list<0/1>
            The list of binary inputs must be the same length as input count.
        parameters  :   list<0/1>
            List of binary values with length equal to gates. Indicates activation of gates.

        Returns
        -------
        list<0/1>
            List of outputs take from the labelled variables given in circuit_description.
        """
        if parameters:
            self.parameters = parameters
        circuit = Circuit(self.circuit_input + self.set_parameters(self.parameters) + self.circuit_output)
        return circuit.run(inputs)

    def set_parameters(self, parameters):
        """
        Takes a set of parameters and modifies the circuit_description gates based on 0 values for gates (changes them to buffer
        first input) returns an updated gate list.

        Parameters
        ----------
        parameters  :   list<0/1>
            List length of gates indicating if a gate is active (1) and no modified or inactive (0) and replaced with
            a buffer on the first input.

        Returns
        -------
        list<list<int, int, int, int>>
            A modified version of the gate list with the correct gates replaced.
        """
        # turn gates off by replacing gates with zero based on parameters
        run_gates = self.gates
        self.parameters = parameters
        for p in range(len(self.parameters)):
            if not int(parameters[p]):
                run_gates[p][2] = 0
            else:
                run_gates[p][2] = 1
        # generate circuit_description with correct gates and input output
        circuit_gates = ""
        for gate in run_gates:
            circuit_gates = circuit_gates + str(gate[0]) + " " + str(gate[1]) + " " + str(gate[2]) + " " \
                            + str(gate[3]) + ", "
        return circuit_gates

    def gen_binary_strings(self):
        """
        Generates the full set of possible binary strings of the same length as the inputs.

        Returns
        -------
        list<String>
            The list of all binary strings of length input count.
        """
        # set up input based on input length as all possible inputs
        binary_values = ["0", "1"]
        for i in range(self.input_count):
            if len(binary_values[0]) == self.input_count:
                break
            binary_values = ["0" + bv for bv in binary_values] + ["1" + bv for bv in binary_values]
        return binary_values
