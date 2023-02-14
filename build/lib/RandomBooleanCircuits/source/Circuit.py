class Circuit:

    gates = ["NONE", "AND", "NAND", "OR", "NOR", "XOR", "NXOR"]

    def __init__(self, circuit_string):
        self.circuitString = circuit_string
        self.inputs = self.circuitString.split(", ")[0].split(" ")
        self.outputs = self.circuitString.split(", ")[-1].split(" ")
        self.gate_set = self.circuitString.split(", ")[1:-1]
        pass

    def run(self, parameters):
        input_dict = {}
        for i in range(len(self.inputs)):
            input_dict[self.inputs[i]] = parameters[i]
        for gate in self.gate_set:
            gate_list = gate.split(" ")
            inputA = int(input_dict[gate_list[0]])
            inputB = int(input_dict[gate_list[1]])
            gate_type = Circuit.gates[int(gate_list[2])]
            output = None
            if gate_type == "AND":
                output = inputA & inputB
            elif gate_type == "NAND":
                output = not(inputA & inputB)
            elif gate_type == "OR":
                output = inputA | inputB
            elif gate_type == "NOR":
                output = not(inputA | inputB)
            elif gate_type == "XOR":
                output = inputA ^ inputB
            elif gate_type == "NXOR":
                output = not(inputA ^ inputB)
            elif gate_type == "NONE":
                output = inputA
            input_dict[gate_list[3]] = output
        output_list = []
        for o in self.outputs:
            output_list.append(int(input_dict[o]))
        return output_list
