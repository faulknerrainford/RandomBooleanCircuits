# noinspection DuplicatedCode
class Circuit:
    """
    Boolean circuits implemented using the representation from: Miller, Julian F., and Others. 1999.
    “An Empirical Study of the Efficiency of Learning Boolean Functions Using a Cartesian Genetic Programming Approach.”
    In Proceedings of the Genetic and Evolutionary Computation Conference, 2:1135–42. researchgate.net.

    In this version we list gates with the following properties:

    0
        Buffer input 1
    1
        AND gate
    2
        NAND gate
    3
        OR gate
    4
        NOR gate
    5
        XOR gate
    6
        NXOR gate

    So a circuit_description is listed as <Inputs> <Gates> <Outputs>

    <Inputs>
        Incremental integer labeling of original inputs.
    <Gates>
        Each gate is a list <Input 1, Input 2, Gate Type, Output> where output is the next incremental
        integer and the gate type is taken from the above list.
    <Outputs>
        Integer list of the values to be read out of the circuit_description.

    Example for a single AND gate circuit_description: [(0, 1) (0, 1, 1, 2) (2)] where 0 and 1 are the
    circuit_description and gate inputs, the gate type is AND=1, and the gate and circuit_description output is 2.

    Parameters
    ----------
    circuit_string  :   String
        Input string of tuples in the <Inputs> <Gates> <Outputs> format.
    """

    gates = ["NONE", "AND", "NAND", "OR", "NOR", "XOR", "NXOR"]

    def __init__(self, circuit_string):
        self.circuitString = circuit_string
        self.inputs = self.circuitString.split(", ")[0].split(" ")
        self.outputs = self.circuitString.split(", ")[-1].split(" ")
        self.gate_set = self.circuitString.split(", ")[1:-1]
        pass

    def run(self, parameters=None):
        """
        Operates the circuit_description on a given set of inputs and returns a list of values from the listed outputs.

        Parameters
        -----------
        parameters  :   List<Integer>
            List of 0 or 1 values representing the value of each input, must be same length as input list. Will default
            to all 1.

        Returns
        --------
        List<Integer>
            List of 0 or 1 values read from the given list of outputs.
        """
        if not parameters:
            parameters = [1 for x in range(len(self.inputs))]
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
