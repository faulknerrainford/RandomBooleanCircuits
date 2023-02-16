"""
:author: Penn Faulkner Rainford
:license: GPL v3
:copyright: 2022-2023
"""

from RandomBooleanCircuits import CircuitExplorer
from RandomBooleanCircuits import fixed_input as fin

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Dict

from pycharc.search.base_ga import (
    GAIndividual, IntGAParameterSpec,
    GAParameterSpec, piecewise_individual, mutate_piecewise, combine_piecewise)
from pycharc.search.microbial_ga import MicrobialGA, KNNFitness
from pycharc.system import System

# reservoirpy.utils.VERBOSITY = 0


"""
RBCParamSpec
------------
Describes the mutable parameters of the system being searched.

Parameters
----------
gate_0  :   IntGAParameterSpec(0,1)
    The gate is active (1) or inactive (0)
gate_1  :   IntGAParameterSpec(0,1)
    The gate is active (1) or inactive (0)
gate_2  :   IntGAParameterSpec(0,1)
    The gate is active (1) or inactive (0)
gate_3  :   IntGAParameterSpec(0,1)
    The gate is active (1) or inactive (0)
gate_4  :   IntGAParameterSpec(0,1)
    The gate is active (1) or inactive (0)
gate_5  :   IntGAParameterSpec(0,1)
    The gate is active (1) or inactive (0)
"""
RBCParamSpec: Dict[str, GAParameterSpec] = {
    'gate_0': IntGAParameterSpec(0, 1),
    'gate_1': IntGAParameterSpec(0, 1),
    'gate_2': IntGAParameterSpec(0, 1),
    'gate_3': IntGAParameterSpec(0, 1),
    'gate_4': IntGAParameterSpec(0, 1),
    'gate_5': IntGAParameterSpec(0, 1),
}


@dataclass
class RBCSystem(System):
    """
    System for running random boolean circuits. Circuit Settings are written as follows:
    Input count: int
        The number of input values being used, these will be labeled indexing from 0 automatically.
    Gate descriptors: list<list<int, int, int, int>>
        The int name of the two inputs followed by the gate type and the label for the gate output.
    Output values: list<ints>
        The list of int names of values to be output.

    In this version we list gate types as follows:
    0   -   Buffer input 1
    1   -   AND gate
    gate_4  :   int
        indicates if gate 4 is active or a buffer for its first input_vals
    gate_5  :   int
        indicates if gate 5 is active or a buffer for its first input_vals
    2   -   NAND gate
    3   -   OR gate
    4   -   NOR gate
    5   -   XOR gate
    6   -   NXOR gate

    Parameters
    ----------
    gate_0  :   int
        indicates if gate 0 is active or a buffer for its first input_vals
    gate_1  :   int
        indicates if gate 1 is active or a buffer for its first input_vals
    gate_2  :   int
        indicates if gate 2 is active or a buffer for its first input_vals
    gate_3  :   int
        indicates if gate 3 is active or a buffer for its first input_vals
    gate_4  :   int
        indicates if gate 4 is active or a buffer for its first input_vals
    gate_5  :   int
        indicates if gate 5 is active or a buffer for its first input_vals
    circuitSettings : list<int, list<list<int>>, list<int>>
        The settings for the circuit_description, [Input count, Gate descriptions, Output values]
    """
    gate_0: int = 1
    gate_1: int = 1
    gate_2: int = 1
    gate_3: int = 1
    gate_4: int = 1
    gate_5: int = 1
    circuitSettings: list = None

    def __post_init__(self):
        """
        Initialises the CircuitExplorer

        """
        super(RBCSystem, self).__post_init__()
        self.circuit = CircuitExplorer(self.circuitSettings[0], self.circuitSettings[1], self.circuitSettings[2])

    def train(self, train_input: npt.NDArray[np.floating], train_output: npt.NDArray[np.floating]) -> None:
        """
        Unused in this instance.
        """
        pass

    def run_one(self, input_vals: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Performs a single run of the system on a single set of input values.

        Parameters
        ----------
        input_vals  :   numpy.NDArray
            array of input values

        Returns
        -------
        Output of circuit_description.run_single.

        """
        return self.circuit.run_single(input_vals, ''.join([str(e) for e in [self.gate_0, self.gate_1, self.gate_2,
                                                                             self.gate_3]]))

    def run(self, inputs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Currently the same as run_one.

        Parameters
        ----------
        inputs  :   numpy.NDArray
            array of input values

        Returns
        -------
        Output of circuit_description.run_single.

        """
        return self.circuit.run_single(inputs, ''.join([str(e) for e in [self.gate_0, self.gate_1, self.gate_2,
                                                                         self.gate_3]]))

    def preprocess(self, input_vals: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        No preprocessing, returns input values without modification

        Parameters
        ----------
        input_vals  : numpy.NDArray
            The input values which might normally be processed

        Returns
        -------
        numpy.NDArray
            The input values without modification
        """
        return input_vals


# RBCIndividual(piecewise_params(RBCParamSpec), RBCParamSpec)
def make_subclass(circuit):
    """
    Dynamic subclassing to define different circuits for multiple runs in the same script.

    Parameters
    -----------
    circuit :   list<int, list<list<int>>, list<int>>
        The circuitSetting for the circuit_description: [input_count, gate_descriptors, output_values]

    Returns
    --------
    RBCIndividual
        The new subclass of GAIndividual with the correct circuit_description settings and system function call.
    """
    class RBCIndividual(GAIndividual):
        cls_circuit = circuit

        @property
        def system(self):
            return RBCSystem(circuitSettings=circuit,
                             input_units=1,
                             output_units=1,
                             hidden_units=1,
                             wash_out=1,
                             gate_0=self['gate_0'],
                             gate_1=self['gate_1'],
                             gate_2=self['gate_2'],
                             gate_3=self['gate_3'],
                             gate_4=self['gate_4'],
                             gate_5=self['gate_5']
                             )  # Or whatever instantiation your system needs

    return RBCIndividual


def circuit_search(settings):
    """
    Takes a text based description of the gates in the circuit_description and sets up the circuit_description settings
    before generating a search object for it using MicrobialGA.

    Parameters
    ----------
    settings    :   list<Strings>
        The list of "AND" and "OR" gates to be implemented in the existing wiring.

    Returns
    --------
    MicrobialGA
        A search object with correct circuit_description and system call.
    """
    input_values = 4
    output_values = [6, 7, 8, 9]
    gate_set = [[0, 1, 1, 4], [2, 3, 2, 5], [0, 5, 2, 6], [1, 6, 1, 7], [2, 7, 1, 8], [3, 8, 2, 9]]
    for j in range(len(settings)):
        if settings[j] == "AND":
            gate_set[j][2] = 1
        elif settings[j] == "OR":
            gate_set[j][2] = 3
        else:
            gate_set[j][2] = 0
    CircuitIndividual = make_subclass([input_values, gate_set, output_values])
    searcher = MicrobialGA(
        metric_spec=[fin.fixed_input_binary_0, fin.fixed_input_binary_1, fin.fixed_input_binary_2,
                     fin.fixed_input_binary_3,
                     fin.fixed_input_binary_4, fin.fixed_input_binary_5, fin.fixed_input_binary_6,
                     fin.fixed_input_binary_7,
                     fin.fixed_input_binary_8, fin.fixed_input_binary_9, fin.fixed_input_binary_10,
                     fin.fixed_input_binary_11, fin.fixed_input_binary_12, fin.fixed_input_binary_13,
                     fin.fixed_input_binary_14, fin.fixed_input_binary_15],
        generator=lambda: piecewise_individual(RBCParamSpec, CircuitIndividual),
        number_of_tests=1,
        population_size=10,
        generations=500,
        mutation_rate=0.1,
        recombination_rate=0.5,
        deme_rate=0.2,
        combine_func=combine_piecewise,
        mutate_func=mutate_piecewise,
        fitness_function=KNNFitness(3)
    )
    return searcher


def add_search_results(settings, results, dataframe):
    """
    Convert search population metrics and final population values to dataframe rows with circuit_description
    description and parameter settings as well as the truth table output and whether a circuit_description is in the
    final population.

    Parameters
    ----------
    settings    :   list<Strings>
        The text based list of "AND" and "OR" values to describe the different gates in the circuit_description.
    results     :   MicrobialGA
        The run MicrobialGA with its final population and general population metrics.
    dataframe   :   pandas.DataFrame
        The dataframe to add the set of results to. Must already be established with correct columns - ["Input_Count",
        "Outputs", "Wiring", "Gates", "Gate_Activations", "Truth_Table", "Final_Population"]

    Returns
    -------
    pandas.DataFrame
        The dataframe with the new data added.
    """
    input_values = 4
    output_values = [6, 7, 8, 9]
    wiring = [[0, 1], [2, 3], [0, 5], [1, 6], [2, 7], [3, 8]]
    final_pop = [p.params for p in results.population]
    for r in results.population_metrics.keys():
        gate_activity = r
        truth_table = results.population_metrics[r]
        if r in final_pop:
            fp = True
        else:
            fp = False
        new_row = {"Input_Count": input_values, "Outputs": output_values, "Wiring": wiring,
                   "Gates": settings, "Gate_Activations": gate_activity, "Truth_Table": truth_table,
                   "Final_Population": fp}
        dataframe = pd.concat([dataframe, pd.DataFrame([new_row])], axis=0, ignore_index=True)
    return dataframe


def save_results(dataframe, filename):
    """
    Save the dataframe of results to a csv file with the given name.

    Parameters
    ----------

    dataframe   :   pandas.DataFrame
        The dataframe of results.
    filename    :   String
        The filename and path to be used to save the file.
    """
    dataframe.to_csv(filename, index=True, columns=["Input_Count", "Outputs", "Wiring", "Gates", "Gate_Activations",
                                                    "Truth_Table", "Final_Population"])
    pass

# set up a loop with modified individual class for each run to loop over all possible circuit_description
#  designs and add results to dataframe after each.


"""
Main functionality running all possible 6 gates with the given wiring using only "AND" and "OR" gates.
"""
if __name__ == "__main__":
    circuit_settings = [
        ["AND", "AND", "AND", "AND", "AND", "AND"],
        ["AND", "AND", "AND", "AND", "AND", "OR"],
        ["AND", "AND", "AND", "AND", "OR", "AND"],
        ["AND", "AND", "AND", "AND", "OR", "OR"],
        ["AND", "AND", "AND", "OR", "AND", "AND"],
        ["AND", "AND", "AND", "OR", "AND", "OR"],
        ["AND", "AND", "AND", "OR", "OR", "AND"],
        ["AND", "AND", "AND", "OR", "OR", "OR"],
        ["AND", "AND", "OR", "AND", "AND", "AND"],
        ["AND", "AND", "OR", "AND", "AND", "OR"],
        ["AND", "AND", "OR", "AND", "OR", "AND"],
        ["AND", "AND", "OR", "AND", "OR", "OR"],
        ["AND", "AND", "OR", "OR", "AND", "AND"],
        ["AND", "AND", "OR", "OR", "AND", "OR"],
        ["AND", "AND", "OR", "OR", "OR", "AND"],
        ["AND", "AND", "OR", "OR", "OR", "OR"],
        ["AND", "OR", "AND", "AND", "AND", "AND"],
        ["AND", "OR", "AND", "AND", "AND", "OR"],
        ["AND", "OR", "AND", "AND", "OR", "AND"],
        ["AND", "OR", "AND", "AND", "OR", "OR"],
        ["AND", "OR", "AND", "OR", "AND", "AND"],
        ["AND", "OR", "AND", "OR", "AND", "OR"],
        ["AND", "OR", "AND", "OR", "OR", "AND"],
        ["AND", "OR", "AND", "OR", "OR", "OR"],
        ["AND", "OR", "OR", "AND", "AND", "AND"],
        ["AND", "OR", "OR", "AND", "AND", "OR"],
        ["AND", "OR", "OR", "AND", "OR", "AND"],
        ["AND", "OR", "OR", "AND", "OR", "OR"],
        ["AND", "OR", "OR", "OR", "AND", "AND"],
        ["AND", "OR", "OR", "OR", "AND", "OR"],
        ["AND", "OR", "OR", "OR", "OR", "AND"],
        ["AND", "OR", "OR", "OR", "OR", "OR"],
        ["OR", "AND", "AND", "AND", "AND", "AND"],
        ["OR", "AND", "AND", "AND", "AND", "OR"],
        ["OR", "AND", "AND", "AND", "OR", "AND"],
        ["OR", "AND", "AND", "AND", "OR", "OR"],
        ["OR", "AND", "AND", "OR", "AND", "AND"],
        ["OR", "AND", "AND", "OR", "AND", "OR"],
        ["OR", "AND", "AND", "OR", "OR", "AND"],
        ["OR", "AND", "AND", "OR", "OR", "OR"],
        ["OR", "AND", "OR", "AND", "AND", "AND"],
        ["OR", "AND", "OR", "AND", "AND", "OR"],
        ["OR", "AND", "OR", "AND", "OR", "AND"],
        ["OR", "AND", "OR", "AND", "OR", "OR"],
        ["OR", "AND", "OR", "OR", "AND", "AND"],
        ["OR", "AND", "OR", "OR", "AND", "OR"],
        ["OR", "AND", "OR", "OR", "OR", "AND"],
        ["OR", "AND", "OR", "OR", "OR", "OR"],
        ["OR", "OR", "AND", "AND", "AND", "AND"],
        ["OR", "OR", "AND", "AND", "AND", "OR"],
        ["OR", "OR", "AND", "AND", "OR", "AND"],
        ["OR", "OR", "AND", "AND", "OR", "OR"],
        ["OR", "OR", "AND", "OR", "AND", "AND"],
        ["OR", "OR", "AND", "OR", "AND", "OR"],
        ["OR", "OR", "AND", "OR", "OR", "AND"],
        ["OR", "OR", "AND", "OR", "OR", "OR"],
        ["OR", "OR", "OR", "AND", "AND", "AND"],
        ["OR", "OR", "OR", "AND", "AND", "OR"],
        ["OR", "OR", "OR", "AND", "OR", "AND"],
        ["OR", "OR", "OR", "AND", "OR", "OR"],
        ["OR", "OR", "OR", "OR", "AND", "AND"],
        ["OR", "OR", "OR", "OR", "AND", "OR"],
        ["OR", "OR", "OR", "OR", "OR", "AND"],
        ["OR", "OR", "OR", "OR", "OR", "OR"],
    ]

    df = pd.DataFrame(columns=["Input_Count", "Outputs", "Wiring", "Gates", "Gate_Activations", "Truth_Table",
                               "Final_Population"])
    df = df.astype({"Final_Population": bool})
    for i in circuit_settings:
        search = circuit_search(i)
        search.run()
        df = add_search_results(i, search, df)

    save_results(df, "~/PycharmProjects/RandomBooleanCircuits/PyCharc_multirun_6_gate_RBC_results.csv")
