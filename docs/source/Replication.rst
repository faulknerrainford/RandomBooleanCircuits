############
Replication
############

Exploring the potential of a Boolean Circuit with Novelty Search
================================================================

In this paper we ran NS over a set of boolean logic circuits with NS able to remove different gates from the circuit to
search for behaviours. This was done on 6 and 12 gate circuits with fixed wirings. The scripts for the NS, data
collected and analysis scripts can all be found at: https://github.com/faulknerrainford/RandomBooleanCircuits

NS scripts
----------

The 6 gate circuit novelty searches can be run from the source directory using::

    python bc_6_gates.py

This runs in under 20 minutes on a intel Core i5 laptop and produces the "PyCharc_multirun_6_gate_RBC_results.csv" file.

Similarly for the 12 gate circuit novelty searchers run from the source directory using::

    python bc_12_gates.py

This runs in ~15 hours on an intel Core i5 laptop and produces the "PyCharc_multirun_12_gate_RBC_results.csv" file.

Data
----

The raw data output from the NS runs can be found for the 6 and 12 gate systems respectively in:
- PyCharc_multirun_6_gate_RBC_results.csv
- PyCharc_multirun_12_gate_RBC_results.csv

The dataframes containing repeated behaviours for both the 6 and 12 gate systems are provided as well:
- 6_var_group_df.csv
- 12_var_group_df.csv

In both cases the larger 12 gate system data is stored compressed in .zip and will need to be extracted before they can
be used for analysis

Analysis
---------

The pycharcAnalysis module is used to produce all analysis and graphs in the paper based on the above data. The main
function of the module will produce based on the number of gates in the system the histogram
of archive size, the box plot of the number of components for each instance of a truth table with multiple instances
found and the scatter graph of the number of truth tables found for each initial circuit. For 6 circuits and 12 circuits
respectively::

    python pycharcAnalysis.py 6
    python pycharcAnalysis.py 12


Graphs
------
Graphs are output with following names based on system with N gates:

N_archive_hist.png
    Histogram of archive size for initial circuits used in NS
N_tt_per_circuits.png
    Scatter graph of number of circuits against the number of unique truth tables found by NS
N_variants_per_tt.png
    Box plot of the component sizes of different circuits which produce the same behaviour across all circuits NS runs.
