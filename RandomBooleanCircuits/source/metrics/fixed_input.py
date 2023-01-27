"""
fixed_input
***********

:author: Penn Faulkner Rainford
:license: GPL v3
:copyright: 2022-2023

"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.linalg import matrix_rank
from pycharc.metrics.registry import register
from pycharc.system import System
from pycharc.sources import random_source, SourceType


def fixed_input(system: System, input: npt.NDArray[np.floating] = (0), output_type="float"):
    # pass to subfuction based on type or run on input_vals and caste return to float.
    if output_type == "binary":
        return fixed_input_binary(system, input)
    else:
        return float(system.run(system.preprocess(input)))
    pass


def fixed_input_binary(system: System, input: npt.NDArray[np.floating]):
    # convert input_vals to correct string using system preprocess
    input = system.preprocess(input)
    output = system.run(input)
    # convert output to float
    output = ''.join(str(i) for i in output)
    return float(int(output, 2))


def fixed_input_binary_0(system: System):
    return fixed_input_binary(system, np.array([0, 0, 0, 0]))


def fixed_input_binary_1(system: System):
    return fixed_input_binary(system, np.array([0, 0, 0, 1]))


def fixed_input_binary_2(system: System):
    return fixed_input_binary(system, np.array([0, 0, 1, 0]))


def fixed_input_binary_3(system: System):
    return fixed_input_binary(system, np.array([0, 0, 1, 1]))


def fixed_input_binary_4(system: System):
    return fixed_input_binary(system, np.array([0, 1, 0, 0]))


def fixed_input_binary_5(system: System):
    return fixed_input_binary(system, np.array([0, 1, 0, 1]))


def fixed_input_binary_6(system: System):
    return fixed_input_binary(system, np.array([0, 1, 1, 0]))


def fixed_input_binary_7(system: System):
    return fixed_input_binary(system, np.array([0, 1, 1, 1]))


def fixed_input_binary_8(system: System):
    return fixed_input_binary(system, np.array([1, 0, 0, 0]))


def fixed_input_binary_9(system: System):
    return fixed_input_binary(system, np.array([1, 0, 0, 1]))


def fixed_input_binary_10(system: System):
    return fixed_input_binary(system, np.array([1, 0, 1, 0]))


def fixed_input_binary_11(system: System):
    return fixed_input_binary(system, np.array([1, 0, 1, 1]))


def fixed_input_binary_12(system: System):
    return fixed_input_binary(system, np.array([1, 1, 0, 0]))


def fixed_input_binary_13(system: System):
    return fixed_input_binary(system, np.array([1, 1, 0, 1]))


def fixed_input_binary_14(system: System):
    return fixed_input_binary(system, np.array([1, 1, 1, 0]))


def fixed_input_binary_15(system: System):
    return fixed_input_binary(system, np.array([1, 1, 1, 1]))
