import sys

import numpy as np
import pandas as pd
import seaborn as sns
import statistics as st
import matplotlib.pyplot as plt
from collections import Counter

from RandomBooleanCircuits import CircuitExplorer


class TruthTableAnalyser:
    """
    Reads in the data from csv file and provides functions for analysis

    Parameters
    ----------
    filename    :   String
        full file path to the data csv from the boolean circuit scripts
    """

    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.truth_to_list()
        # self.missing_fc_add()
        self.gates_to_label()

    def fc_in_fp(self, inputs=None, outputs=None, gates=None):
        """
        Finds if the full circuit for each gate set is in the final population.

        Parameters
        -----------
        inputs  :   List<Int>
            The set of inputs to the circuit, used to identify correct circuit
        outputs :   List<Int>
            The set of outputs from the circuit, used to identify correct circuit
        gates   :   List<String>
            The set of gate in the circuits, used to identify correct circuit

        Return
        ------
        DataFrame
            Group by with the truth value for if the data for each circuit contains the full circuit
            in the final population
        """
        if "Comp_Count" not in self.df.columns:
            self.add_comp_count()
        if inputs:
            group = self.df.loc[self.df["Input_Count"] == inputs]
        else:
            group = self.df
        if outputs:
            group = group.loc[group["Outputs"] == outputs]
        if gates:
            group = group.loc[group["Gates"] == gates]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.find_fc)

    @staticmethod
    def find_fc(df):
        """
        Returns truth value for if the full circuit is in the final population.

        Parameter
        ---------
        df  :   DataFrame
            dataframe of the circuit to check for full circuit.

        Returns
        -------
        Boolean
            true if circuit with full component count in the final population, false otherwise.
        """
        if df.loc[(df["Comp_Count"] == GATES), "Final_Population"].item():
            return True
        else:
            return False

    @staticmethod
    def find_fc_tt(df):
        """
        Checks if the truth table for the full circuit is in the final population.

        Parameter
        ---------
        df  :   DataFrame
            Dataframe to search for the full circuit.

        Returns
        -------
        Boolean
            The truth value for if the truth table of the full circuit is in the final population.
        """
        if df.loc[(df["Comp_Count"] == GATES), "Final_Population"].item():
            return True
        else:
            tt = df.loc[(df["Comp_Count"] == 4), "Truth_Table"].item()
            if any([tt == x for x in df.loc[(df["Final_Population"]), "Truth_Table"].values]):
                return True
            else:
                return False

    def fc_tt_in_fp(self, inputs=None, outputs=None, gates=None):
        """
        Checks for the full circuit truth table in the final population

        Parameters
        ----------
        inputs  :   List<Int>
            The set of inputs to the circuit, used to identify correct circuit
        outputs :   List<Int>
            The set of outputs from the circuit, used to identify correct circuit
        gates   :   List<String>
            The set of gate in the circuits, used to identify correct circuit]

        Returns
        -------
        DataFrame
            groupby of truth values for if the full circuits truth table is in the final population.
        :return:
        """
        if "Comp_Count" not in self.df.columns:
            self.add_comp_count()
        if inputs:
            group = self.df.loc[self.df["Input_Count"] == inputs]
        else:
            group = self.df
        if outputs:
            group = group.loc[group["Outputs"] == outputs]
        if gates:
            group = group.loc[group["Gates"] == gates]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.find_fc_tt)

    def av_dist_fc_to_fp(self):
        """
        Gets average hamming distance of final population parameters to full circuit.

        Returns
        -------
        DataFrame
            groupby with average hamming distance for each final population.
        """
        if "FC_Dist" not in self.df.columns:
            self.add_fc_dist()
        return self.df.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["FC_Dist"].mean()

    def av_tt_dist_fc_to_fp(self):
        """
        Gets average hamming distance of truth tables of the final population to the full circuit.

        Returns
        -------
        DataFrame
            groupby with mean hamming distance for each final population.
        :return:
        """
        if "FC_TT_Dist" not in self.df.columns:
            self.add_fc_tt_dist()
        return self.df.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["FC_TT_Dist"].mean()

    def av_dist_btw_fp(self):
        """
        Average hamming distance between the parameters of the final population

        Returns
        -------
        DataFrame
            groupby of mean hamming distances between the parameters of the final population
        """
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.calc_dist_circuit)

    def av_tt_dist_btw_fp(self):
        """
        Average hamming distance between the truth tables of the final population

        Returns
        -------
        DataFrame
            groupby of mean hamming distances between the truth tables of the final population
        """
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.calc_tt_dist_circuit)

    def max_dist_btw_fp(self):
        """
        Max hamming distance between the parameters of the final population

        Returns
        -------
        DataFrame
            groupby of max hamming distances between the parameters of the final population
        """
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.calc_max_dist_circuit)

    def max_tt_dist_btw_fp(self):
        """
        Max hamming distance between the truth tables of the final population

        Returns
        -------
        DataFrame
            groupby of max hamming distances between the truth tables of the final population
        """
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.calc_max_tt_dist_circuit)

    def max_comp_in_fp(self):
        """
        Max components in the final population

        Returns
        -------
        DataFrame
            groupby of max parameters of the final population
        """
        self.add_comp_count()
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["Comp_Count"].agg(max)

    def min_comp_in_fp(self):
        """
        Min components in the final population

        Returns
        -------
        DataFrame
            groupby of min parameters of the final population
        """
        self.add_comp_count()
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["Comp_Count"].agg(min)

    def mode_comp_in_fp(self):
        """
        Mode components in the final population

        Returns
        -------
        DataFrame
            groupby of mode parameters of the final population
        """
        self.add_comp_count()
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["Comp_Count"].agg(st.mode)

    def median_comp_in_fp(self):
        """
        Median components in the final population

        Returns
        -------
        DataFrame
            groupby of median parameters of the final population
        """
        self.add_comp_count()
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["Comp_Count"].agg(st.median)

    def add_comp_count(self):
        """
        Adds "Comp_Count" column to dataframe containing the count of the number of active gates in the circuit.
        """
        self.df["Comp_Count"] = [x.count("1") for x in self.df["Gate_Activations"]]

    def add_comp_fraction(self):
        """
        Adds "Comp_Frac" column to dataframe containing the proportion of AND gates in the gate set.
        """
        self.df["Comp_Frac"] = [x.count("AND") / (x.count("OR") + x.count("AND")) for x in self.df["Gates"]]

    def add_fc_dist(self):
        """
        Adds "FC_Dist" column to dataframe containing the number of inactive gates.
        """
        self.df["FC_Dist"] = [x.count("0") for x in self.df["Gate_Activations"]]

    def add_fc_tt_dist(self):
        """
        Adds "FC_TT_Dist" column to dataframe containing the hamming distance between the truth table and
        the full circuits truth table.
        """
        # get set of circuits in frame
        tt_dict = {}
        for i in range(self.df.shape[0]):
            key = tuple([self.df.iloc[i]["Input_Count"], self.df.iloc[i]["Outputs"],
                         self.df.iloc[i]["Wiring"], self.df.iloc[i]["Gates"]])
            if key not in tt_dict.keys():
                tt_dict[key] = None
        # get id for each frame
        for k in tt_dict.keys():
            index_fc = self.df.index[(self.df["Input_Count"] == k[0]) & (self.df["Outputs"] == k[1]) &
                                     (self.df["Wiring"] == k[2]) & (self.df["Gates"] == k[3]) &
                                     (self.df["Gate_Activations"].str.count("0") == 0)].tolist()
            tt_dict[k] = index_fc[0]
        self.df["FC_TT_Dist"] = [self.calc_tt_dist_2(ind, tt_dict[tuple([self.df.iloc[ind]["Input_Count"],
                                                                         self.df.iloc[ind]["Outputs"],
                                                                         self.df.iloc[ind]["Wiring"],
                                                                         self.df.iloc[ind]["Gates"]])])
                                 for ind in range(self.df.shape[0])]

    def missing_fc_add(self):
        """
        Adds the full circuit with truth table to the dataframe if not already present.
        """
        tt_dict = {}
        for i in range(self.df.shape[0]):
            key = tuple([self.df.iloc[i]["Input_Count"], self.df.iloc[i]["Outputs"],
                         self.df.iloc[i]["Wiring"], self.df.iloc[i]["Gates"]])
            if key not in tt_dict.keys():
                tt_dict[key] = None
        for k in tt_dict.keys():
            index_fc = self.df.index[(self.df["Input_Count"] == k[0]) & (self.df["Outputs"] == k[1]) &
                                     (self.df["Wiring"] == k[2]) & (self.df["Gates"] == k[3]) &
                                     (self.df["Gate_Activations"].str.count("0") == 0)].tolist()
            if not index_fc:
                tt_dict[k] = self.get_fc_tt_index(k[0], k[1], k[2], k[3])

    def get_fc_tt_index(self, input_count, outputs, wiring, gates):
        """
        For a given circuit add full circuit to the dataframe and gets the index.

        Parameters
        ----------
        input_count :   Int
            Number of inputs into the circuit, used for circuit identification.
        outputs     :   List<Int>
            List of outputs from the circuit, used for circuit identification.
        wiring      :   List<List<Int, Int>>
            List of gate input lists, used for circuit identification.
        gates       :   List<String>
            List of gates in circuits, used for circuit identification.

        Returns
        -------
        Int
            Index of full circuit
        """
        # generate circuit_description
        gate_list = []
        output_label = input_count
        gates = gates.strip('][').split(', ')
        wiring = wiring.strip('][').split('], [')
        outputs = outputs.strip('][').split(', ')
        wiring = [[int(x[0]), int(x[3])] for x in wiring]
        for gate, wires in zip(gates, wiring):
            if gate == "'AND'":
                wires = wires + [1]
            elif gate == "'OR'":
                wires = wires + [3]
            else:
                wires = wires + [0]
            wires = wires + [output_label]
            gate_list.append(wires)
            output_label = output_label + 1
        circuit = CircuitExplorer(inputs=input_count, gates=gate_list, outputs=outputs)
        # get circuit_description tt
        parameters = ""
        parameters_string = "("
        for _ in gate_list:
            parameters = parameters + '1'
            parameters_string = parameters_string + '1, '
        parameters_string = parameters_string[:-2] + ")"
        tt = circuit.run(parameters)
        tt = [float(int(''.join(str(j) for j in i), 2)) for i in tt]
        # add results to dataframe
        new_row = {"Input_Count": input_count, "Outputs": str([int(output) for output in outputs]),
                   "Wiring": str(wiring), "Gates": str([gate[1:-1] for gate in gates]),
                   "Gate_Activations": parameters_string, "Truth_Table": tt, "Final_Population": False}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], axis=0, ignore_index=True)
        # get index of new row
        index_fc = self.df.shape[0] - 1
        return index_fc

    def calc_dist_2(self, index_a, index_b, df=None):
        """
        Calculates the hamming distance between the parameters of two circuits given by the indexes in the dataframe.

        Parameters
        ----------
        index_a :   Int
            Index of the first circuit
        index_b :   Int
            Index of the second circuit
        df      :   DataFrame
            dataframe for selecting circuits from subset.

        Returns
        -------
        Int
            The hamming distance between the parameters of the two circuits.
        """
        if df is None:
            df = self.df
        dist = 0
        for (a, b) in zip(df.at[index_a, "Gate_Activations"], df.at[index_b, "Gate_Activations"]):
            if a != b:
                dist = dist + 1
        return dist

    def calc_dist_group(self, inputs=None, outputs=None, gates=None, final=False):
        """
        Calculates the mean hamming distance between parameters for a group of circuits.

        Parameters
        ----------
        inputs  :   Int
            Number of inputs into the circuit, used for group identification
        outputs :   List<Int>
            List of outputs from the circuit, used for group identification
        gates   :   List<String>
            List of gates in the circuit, used for group identification
        final   :   Boolean
            Select only the final population (true) or full archive (false)

        Returns
        -------
        float
            The mean hamming distance between the parameters of circuits in the group.
        """
        if inputs:
            group = self.df.loc[self.df["Input_Count"] == inputs]
        else:
            group = self.df
        if outputs:
            group = group.loc[group["Outputs"] == outputs]
        if gates:
            group = group.loc[group["Gates"] == gates]
        if final:
            group = group.loc[group["Final_Population"]]
        dists = []
        for ind_a in group.index:
            for ind_b in [b for b in group.index if b > ind_a]:
                dists.append(self.calc_dist_2(ind_a, ind_b, group))
        return np.mean(dists)

    def calc_dist_circuit(self, df):
        """
        For a dataframe get the mean hamming distance between parameters in all circuits.

        Parameters
        ----------
        df  :   DataFrame
            set of circuits to find average distance between.

        Returns
        -------
        float
            mean hamming distance between parameters of circuits.
        """
        dists = []
        for ind_a in df.index:
            for ind_b in [b for b in df.index if b > ind_a]:
                dists.append(self.calc_dist_2(ind_a, ind_b, df))
        return np.mean(dists)

    def calc_max_dist_circuit(self, df):
        """
        For a dataframe get the max hamming distance between parameters in all circuits.

        Parameters
        ----------
        df  :   DataFrame
            set of circuits to find max distance between.

        Returns
        -------
        float
            max hamming distance between parameters of circuits.
        """
        dists = []
        for ind_a in df.index:
            for ind_b in [b for b in df.index if b > ind_a]:
                dists.append(self.calc_dist_2(ind_a, ind_b, df))
        return np.max(dists)

    def calc_tt_dist_2(self, index_a, index_b, df=None):
        """
        Calculates the hamming distance between the truth tables of two circuits given by the indexes in the dataframe.

        Parameters
        ----------
        index_a :   Int
            Index of the first circuit
        index_b :   Int
            Index of the second circuit
        df      :   DataFrame
            dataframe for selecting circuits from subset.

        Returns
        -------
        Int
            The hamming distance between the truth tables of the two circuits.
        """
        if df is None:
            df = self.df
        dist = 0
        for (a, b) in zip(df.at[index_a, "Truth_Table"], df.at[index_b, "Truth_Table"]):
            if a != b:
                dist = dist + 1
        return dist

    def calc_tt_dist_circuit(self, df):
        """
        For a dataframe get the mean hamming distance between truth tables in all circuits.

        Parameters
        ----------
        df  :   DataFrame
            set of circuits to find average distance between.

        Returns
        -------
        float
            mean hamming distance between the truth tables of circuits.
        """
        dists = []
        for ind_a in df.index:
            for ind_b in [b for b in df.index if b > ind_a]:
                dists.append(self.calc_tt_dist_2(ind_a, ind_b, df))
        return np.mean(dists)

    def calc_max_tt_dist_circuit(self, df):
        """
        For a dataframe get the max hamming distance between truth tables in all circuits.

        Parameters
        ----------
        df  :   DataFrame
            set of circuits to find max distance between.

        Returns
        -------
        float
            max hamming distance between the truth tables of circuits.
        """
        dists = []
        for ind_a in df.index:
            for ind_b in [b for b in df.index if b > ind_a]:
                dists.append(self.calc_tt_dist_2(ind_a, ind_b, df))
        return np.max(dists)

    def truth_to_list(self):
        """
        Converts the string based truth tables to lists
        """
        self.df["Truth_Table"] = [x.strip("[]").split(". ") for x in self.df["Truth_Table"]]
        self.df["Truth_Table"] = [[int(n.strip(".")) for n in x] for x in self.df["Truth_Table"]]

    def gates_to_label(self):
        """
        Converts the gate set to a base 10 label and adds new column of them, "Labels"
        """
        labels = []
        for gates in self.df["Gates"]:
            label = ''
            gate_list = gates.strip('][').split(', ')
            for gate in gate_list:
                if gate == "'AND'":
                    label = label + '0'
                else:
                    label = label + '1'
            labels.append(int(label, 2))
        self.df["Labels"] = labels

    def tt_distance_graph(self):
        """
        Returns a boxplot of the component counts against the distance of circuits to the full circuit truth tables.

        Returns
        -------
        pyplot
            seaborn box plot
        """
        self.add_comp_count()
        self.add_fc_tt_dist()
        self.add_fc_dist()
        self.add_comp_fraction()
        sns.boxplot(data=self.df, x="Comp_Frac", y="FC_TT_Dist")
        return sns.boxplot(data=self.df, x="Comp_Count", y="FC_TT_Dist")

    def comp_distance_graph(self):
        """
        Returns a boxplot of the component counts against the distance of circuits to the full circuit parameters.

        Returns
        -------
        pyplot
            seaborn box plot
        """
        self.add_comp_count()
        self.add_fc_tt_dist()
        self.add_fc_dist()
        self.add_comp_fraction()
        sns.boxplot(data=self.df, x="Comp_Frac", y="FC_Dist")
        return sns.boxplot(data=self.df, x="Comp_Count", y="FC_Dist")

    def circuit_graph_tt_line(self):
        """
        Returns a line plot of circuit labels against the average truth tables distances between the full population.

        Returns
        -------
        pyplot
            seaborn line plot
        """
        self.add_comp_count()
        self.add_fc_tt_dist()
        self.add_fc_dist()
        self.add_comp_fraction()
        return sns.lineplot(x=self.get_group_labels(), y=self.av_tt_dist_btw_fp())

    def circuit_graph_tt_max_line(self):
        """
        Returns a line plot of circuit labels against the max truth tables distances between the full population.

        Returns
        -------
        pyplot
            seaborn line plot
        """
        self.add_comp_count()
        self.add_fc_tt_dist()
        self.add_fc_dist()
        self.add_comp_fraction()
        return sns.lineplot(x=self.get_group_labels(), y=self.max_tt_dist_btw_fp(), color='r')

    def get_group_labels(self):
        """
        Assigns and returns a list of circuit labels for all final population members.

        Returns
        -------
        Series
            List of labels for circuits.
        """
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["Labels"].agg(max)

    def comp_size_by_tt(self):
        """
        Returns a violin plot of truth table labels against the component counts.

        Returns
        -------
        pyplot
            seaborn violin plot
        """
        # sort tt labels and return comp size box plot (bar style for compression)
        self.add_comp_count()
        fcs = [tuple(x) for x in self.df["Truth_Table"]]
        tts = []
        tts_labels = {}
        tt_label = []
        for i in range(len(fcs)):
            if list(fcs[i]) not in tts:
                tts.append(list(fcs[i]))
                tts_labels[tuple(fcs[i])] = i
                tt_label.append(i)
            else:
                tt_label.append(tts_labels[tuple(fcs[i])])
        self.df["Truth_Label"] = tt_label
        # get only truth tables with multiple instances
        group = self.df[self.df.duplicated("Truth_Table", keep=False).values]
        sns.violinplot(group, x="Truth_Label", y="Comp_Count", cut=True)

    def variants_by_fc(self):
        """
        Returns truth tables violin plots for each full circuit_description truth table showing the component counts.

        Returns
        -------
        ax
            Matplotlib Axes with violin plot of the component count of versions of each circuit's truth table found in
            the full set of runs.
        """
        # get list of unique fc tts
        self.add_comp_count()
        fcs = [tuple(x) for x in self.df.loc[(self.df["Gate_Activations"].str.count("0") == 0), "Truth_Table"]]
        tts = []
        tts_labels = {}
        for i in range(len(fcs)):
            if fcs not in tts:
                tts.append(list(fcs[i]))
                tts_labels[tuple(fcs[i])] = i
        # reduce df to rows with those tts
        tt_df = self.df.loc[(self.df["Truth_Table"].isin(tts))]
        # then plot violin with tt on x and comp count on y.
        tt_df["TT_Labels"] = [tts_labels[tuple(x)] for x in tt_df["Truth_Table"]]
        return sns.violinplot(tt_df, x="TT_Labels", y="Comp_Count", cut=0)

    def get_archive_size(self):
        """
        Get histogram of archive sizes for different initial circuits.

        Returns
        -------
        pyplot
            seaborn histogram
        """
        df_arch_count = self.df.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).size()
        return sns.histplot(df_arch_count)

    def truth_cc_boxplot(self):
        """
        Plots box plot of the truth labels (unique truth tables) against component counts.

        Returns
        -------
        pyplot
            seaborn box plot
        """
        # See boxplot above.
        self.add_comp_count()
        # group = self.df.loc[(self.df["Final_Population"])]
        group = self.df
        fcs = [tuple(x) for x in group["Truth_Table"]]
        tts = []
        tts_labels = {}
        tt_label = []
        for i in range(len(fcs)):
            if list(fcs[i]) not in tts:
                tts.append(list(fcs[i]))
                tts_labels[tuple(fcs[i])] = i
                tt_label.append(i)
            else:
                tt_label.append(tts_labels[tuple(fcs[i])])
        group["Truth_Label"] = tt_label
        return sns.boxplot(group, x="Truth_Label", y="Comp_Count")

    def tt_by_circuit(self):
        """
        scatter plot of the number of truth tables found for each circuit.

        Returns
        -------
        pyplot
            seaborn scatter plot
        """
        # group by circuit_description and assign incremental label and count of unique tts
        # group = self.df.loc[(self.df["Final_Population"])]
        group = self.df
        group_tt_count = Counter(group.groupby(["Input_Count", "Outputs",
                                                "Gates", "Wiring"]).apply(self.unique_tt).values)
        return sns.scatterplot(x=group_tt_count.keys(), y=group_tt_count.values())

    @staticmethod
    def unique_tt(df):
        """
        Count of unique truth tables in the given dataframe.

        Parameters
        ----------
        df  :   DataFrame
            dataframe to be checked for unique truth tables.

        Returns
        -------
        Int
            count of unique truth tables
        """
        tables = [tuple(x) for x in df["Truth_Table"]]
        tables = set(tables)
        return len(list(tables))

    @staticmethod
    def get_label(df):
        """
        Gets the first label value for the given dataframe.

        Parameters
        ----------
        df  :   DataFrame
            dataframe to get label value from
        Returns
        -------
        String
            first Label value
        """
        return df.Labels.unique()[0]


def get_behaviour_table(df_name):
    """
    Prints a fully formated latex table of truth tables.

    Parameters
    ----------
    df_name :   String
        file name for csv to print truth tables from.
    """
    df = pd.read_csv(df_name)
    table_string = "Behaviour Label"
    input_strings = ["0000", "0001", "0010", "0011",
                     "0100", "0101", "0110", "0111",
                     "1000", "1001", "1010", "1011",
                     "1100", "1101", "1110", "1111"]
    behaviour_labels = []
    truth_tables = [x.strip("[]").split(", ") for x in df["Truth_Table"]]
    truth_tables = [[int(n.strip(".")) for n in x] for x in truth_tables]
    behaviours = []
    for x in range(len(truth_tables)):
        if int(df["Truth_Label"][x]) not in behaviour_labels:
            behaviour_labels.append(int(df["Truth_Label"][x]))
            behaviours.append(truth_tables[x])
    for i in range(len(behaviour_labels)):
        table_string = table_string + " & " + str(behaviour_labels[i])
    for i in range(16):
        table_string = table_string + " \\\\ " + input_strings[i]
        for j in range(len(behaviours)):
            table_string = table_string + " & " + str(behaviours[j][i])
    print(table_string)


"""
Main behaviour produces archive size histogram, truth table by circuit scatter plot and component count for truth table
boxplot for paper data and table of behaviours for repeated truth tables.

Parameters
-----------
Gates(Optional) 
    Gives the size of system to analyse.
"""
if __name__ == "__main__":
    if len(sys.argv) > 1:
        GATES = int(sys.argv[1])
    else:
        GATES = 6
    analyser = TruthTableAnalyser("~/PycharmProjects/RandomBooleanCircuits/PyCharc_multirun_" + str(GATES)
                                  + "_gate_RBC_results.csv")
    ax = analyser.get_archive_size()
    plt.ylim(0, 18)
    ax.set_xlabel("Archive Size")
    ax.set_xlabel("No. Circuits")
    plt.show()
    plt.savefig(str(GATES) + "_archive_hist.png")
    analyser.tt_by_circuit()
    plt.savefig(str(GATES) + "_tt_per_circuits.png")
    plt.show()
    plt.figure(figsize=(9, 4))
    ax = analyser.truth_cc_boxplot()
    ax.set_xlabel("Behaviour Label")
    ax.set_ylabel("Component Count")
    plt.savefig(str(GATES) + "_variants_per_tt.png")
    plt.show()
    get_behaviour_table("~/PycharmProjects/RandomBooleanCircuits/" + str(GATES)
                        + "_var_group_df.csv")
    pass
