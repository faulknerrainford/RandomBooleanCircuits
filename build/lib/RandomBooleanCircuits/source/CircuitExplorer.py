from RandomBooleanCircuits import Circuit


class CircuitExplorer:

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
        # turn gates off by replacing gates with zero based on parameters
        circuit = Circuit(self.circuit_input + self.set_parameters(parameters) + self.circuit_output)
        # run all inputs on circuit_description and return an ordered truth table
        truth_table = []
        for bit_string in self.input_values:
            truth_table.append(circuit.run(bit_string))
        return truth_table

    def run_single(self, inputs, parameters=None):
        if parameters:
            self.parameters = parameters
        circuit = Circuit(self.circuit_input + self.set_parameters(self.parameters) + self.circuit_output)
        return circuit.run(inputs)

    def set_parameters(self, parameters):
        # turn gates off by replacing gates with zero based on parameters
        run_gates = self.gates
        self.parameters = parameters
        for p in range(len(self.parameters)):
            if not int(parameters[p]):
                run_gates[p][2] = 0
        # generate circuit_description with correct gates and input output
        circuit_gates = ""
        for gate in run_gates:
            circuit_gates = circuit_gates + str(gate[0]) + " " + str(gate[1]) + " " + str(gate[2]) + " " \
                            + str(gate[3]) + ", "
        return circuit_gates

    def gen_binary_strings(self):
        # set up input based on input length as all possible inputs
        binary_values = ["0", "1"]
        for i in range(self.input_count):
            if len(binary_values[0]) == self.input_count:
                break
            binary_values = ["0" + bv for bv in binary_values] + ["1" + bv for bv in binary_values]
        return binary_values
