import numpy as np
import pandas as pd
import seaborn as sns
import statistics as st
import matplotlib.pyplot as plt
from collections import Counter

from RandomBooleanCircuits import CircuitExplorer


class TruthTableAnalyser:

    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.truth_to_list()
        self.missing_fc_add()
        self.gates_to_label()

    def fc_in_fp(self, inputs=None, outputs=None, gates=None):
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
        if df.loc[(df["Comp_Count"] == 4), "Final_Population"].item():
            return True
        else:
            return False

    @staticmethod
    def find_fc_tt(df):
        if df.loc[(df["Comp_Count"] == 4), "Final_Population"].item():
            return True
        else:
            tt = df.loc[(df["Comp_Count"] == 4), "Truth_Table"].item()
            if any([tt == x for x in df.loc[(df["Final_Population"]), "Truth_Table"].values]):
                return True
            else:
                return False

    def fc_tt_in_fp(self, inputs=None, outputs=None, gates=None):
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
        if "FC_Dist" not in self.df.columns:
            self.add_fc_dist()
        return self.df.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["FC_Dist"].mean()

    def av_tt_dist_fc_to_fp(self):
        if "FC_TT_Dist" not in self.df.columns:
            self.add_fc_tt_dist()
        return self.df.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["FC_TT_Dist"].mean()

    def av_dist_btw_fp(self):
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.calc_dist_circuit)

    def av_tt_dist_btw_fp(self):
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.calc_tt_dist_circuit)

    def max_dist_btw_fp(self):
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.calc_max_dist_circuit)

    def max_tt_dist_btw_fp(self):
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.calc_max_tt_dist_circuit)

    def max_comp_in_fp(self):
        self.add_comp_count()
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["Comp_Count"].agg(max)

    def min_comp_in_fp(self):
        self.add_comp_count()
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["Comp_Count"].agg(min)

    def mode_comp_in_fp(self):
        self.add_comp_count()
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["Comp_Count"].agg(st.mode)

    def median_comp_in_fp(self):
        self.add_comp_count()
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["Comp_Count"].agg(st.median)

    def add_comp_count(self):
        self.df["Comp_Count"] = [x.count("1") for x in self.df["Gate_Activations"]]

    def add_comp_fraction(self):
        self.df["Comp_Frac"] = [x.count("AND") / (x.count("OR") + x.count("AND")) for x in self.df["Gates"]]

    def add_fc_dist(self):
        self.df["FC_Dist"] = [x.count("0") for x in self.df["Gate_Activations"]]

    def add_fc_tt_dist(self):
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
        if df is None:
            df = self.df
        dist = 0
        for (a, b) in zip(df.at[index_a, "Gate_Activations"], df.at[index_b, "Gate_Activations"]):
            if a != b:
                dist = dist + 1
        return dist

    def calc_dist_group(self, inputs=None, outputs=None, gates=None, final=False):
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
        dists = []
        for ind_a in df.index:
            for ind_b in [b for b in df.index if b > ind_a]:
                dists.append(self.calc_dist_2(ind_a, ind_b, df))
        return np.mean(dists)

    def calc_max_dist_circuit(self, df):
        dists = []
        for ind_a in df.index:
            for ind_b in [b for b in df.index if b > ind_a]:
                dists.append(self.calc_dist_2(ind_a, ind_b, df))
        return np.max(dists)

    def calc_tt_dist_2(self, index_a, index_b, df=None):
        if df is None:
            df = self.df
        dist = 0
        for (a, b) in zip(df.at[index_a, "Truth_Table"], df.at[index_b, "Truth_Table"]):
            if a != b:
                dist = dist + 1
        return dist

    def calc_tt_dist_circuit(self, df):
        dists = []
        for ind_a in df.index:
            for ind_b in [b for b in df.index if b > ind_a]:
                dists.append(self.calc_tt_dist_2(ind_a, ind_b, df))
        return np.mean(dists)

    def calc_max_tt_dist_circuit(self, df):
        dists = []
        for ind_a in df.index:
            for ind_b in [b for b in df.index if b > ind_a]:
                dists.append(self.calc_tt_dist_2(ind_a, ind_b, df))
        return np.max(dists)

    def truth_to_list(self):
        self.df["Truth_Table"] = [x.strip("[]").split(". ") for x in self.df["Truth_Table"]]
        self.df["Truth_Table"] = [[int(n.strip(".")) for n in x] for x in self.df["Truth_Table"]]

    def gates_to_label(self):
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
        self.add_comp_count()
        self.add_fc_tt_dist()
        self.add_fc_dist()
        self.add_comp_fraction()
        sns.boxplot(data=self.df, x="Comp_Frac", y="FC_TT_Dist")
        return sns.boxplot(data=self.df, x="Comp_Count", y="FC_TT_Dist")

    def comp_distance_graph(self):
        self.add_comp_count()
        self.add_fc_tt_dist()
        self.add_fc_dist()
        self.add_comp_fraction()
        sns.boxplot(data=self.df, x="Comp_Frac", y="FC_Dist")
        return sns.boxplot(data=self.df, x="Comp_Count", y="FC_Dist")

    def circuit_graph_tt_line(self):
        self.add_comp_count()
        self.add_fc_tt_dist()
        self.add_fc_dist()
        self.add_comp_fraction()
        return sns.lineplot(x=self.get_group_labels(), y=self.av_tt_dist_btw_fp())

    def circuit_graph_tt_max_line(self):
        self.add_comp_count()
        self.add_fc_tt_dist()
        self.add_fc_dist()
        self.add_comp_fraction()
        return sns.lineplot(x=self.get_group_labels(), y=self.av_tt_dist_btw_fp(), color='r')

    def get_group_labels(self):
        group = self.df.loc[self.df["Final_Population"]]
        return group.groupby(["Input_Count", "Outputs", "Gates", "Wiring"])["Labels"].agg(max)

    def comp_size_by_tt(self):
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

    def three_point_circuit_dist(self):
        # chart the max tt dist in fp, mean and min on single graph as bars for each circuit_description.
        # See boxplot above.
        self.add_comp_count()
        group = self.df.loc[(self.df["Final_Population"])]
        # May need new df of tt distances with pairs of index and labels.
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
        return sns.violinplot(group, x="Truth_Label", y="Comp_Count", cut=0)

    def tt_by_circuit(self):
        # group by circuit_description and assign incremental label and count of unique tts
        group = self.df.loc[(self.df["Final_Population"])]
        group_tt_count = Counter(group.groupby(["Input_Count", "Outputs",
                                                "Gates", "Wiring"]).apply(self.unique_tt).values)
        # group_label = self.df.groupby(["Input_Count", "Outputs", "Gates", "Wiring"]).apply(self.get_label)
        # graph as scatter
        return sns.scatterplot(x=group_tt_count.keys(), y=group_tt_count.values())

    @staticmethod
    def unique_tt(df):
        tables = [tuple(x) for x in df["Truth_Table"]]
        return len(list(set(tables)))

    @staticmethod
    def get_label(df):
        return df.Labels.unique()[0]


if __name__ == "__main__":
    GATES = 6
    analyser = TruthTableAnalyser("~/PycharmProjects/RandomBooleanCircuits/PyCharc_multirun_" + str(GATES)
                                  + "_gate_RBC_results.csv")
    analyser.tt_by_circuit()
    plt.savefig(str(GATES) + "_tt_per_circuits.png")
    plt.show()
    # analyser.three_point_circuit_dist()
    # plt.savefig(str(GATES) + "_variants_per_tt.png")
    # plt.show()
    pass
