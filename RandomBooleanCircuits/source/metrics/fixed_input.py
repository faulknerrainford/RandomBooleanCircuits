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
    """
    Takes the system, input and type and passes to subfunction based on type.

    Parameters
    -----------
    system      :   System
        PyCharc system
    input       :   npt.NDArray[np.floating]
        input for the metric
    output_type :   string
        output type needed

    Return
    --------
    float
        the value of the metric
    """
    # pass to subfuction based on type or run on input_vals and caste return to float.
    if output_type == "binary":
        return fixed_input_binary(system, input)
    else:
        return float(system.run(system.preprocess(input)))
    pass


def fixed_input_binary(system: System, input: npt.NDArray[np.floating]):
    """
    Takes the system and the binary input to feed into the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system
    input   :   npt.NDArray[np.floating]
        input for metric to run on

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    # convert input_vals to correct string using system preprocess
    input = system.preprocess(input)
    output = system.run(input)
    # convert output to float
    output = ''.join(str(i) for i in output)
    return float(int(output, 2))


def fixed_input_binary_0(system: System):
    """
    Takes the system and feeds in 0000 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([0, 0, 0, 0]))


def fixed_input_binary_1(system: System):
    """
    Takes the system and feeds in 0001 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([0, 0, 0, 1]))


def fixed_input_binary_2(system: System):
    """
    Takes the system and feeds in 0010 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([0, 0, 1, 0]))


def fixed_input_binary_3(system: System):
    """
    Takes the system and feeds in 0011 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([0, 0, 1, 1]))


def fixed_input_binary_4(system: System):
    """
    Takes the system and feeds in 0100 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([0, 1, 0, 0]))


def fixed_input_binary_5(system: System):
    """
    Takes the system and feeds in 0101 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([0, 1, 0, 1]))


def fixed_input_binary_6(system: System):
    """
    Takes the system and feeds in 0110 the system and returns the float converted output.

    Parameters




    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([0, 1, 1, 0]))


def fixed_input_binary_7(system: System):
    """
    Takes the system and feeds in 0111 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([0, 1, 1, 1]))


def fixed_input_binary_8(system: System):
    """
    Takes the system and feeds in 1000 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([1, 0, 0, 0]))






def fixed_input_binary_9(system: System):
    """
    Takes the system and feeds in 1001 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([1, 0, 0, 1]))


def fixed_input_binary_10(system: System):
    """
    Takes the system and feeds in 1010 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([1, 0, 1, 0]))


def fixed_input_binary_11(system: System):
    """
    Takes the system and feeds in 1011 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([1, 0, 1, 1]))


def fixed_input_binary_12(system: System):
    """
    Takes the system and feeds in 1100 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([1, 1, 0, 0]))


def fixed_input_binary_13(system: System):
    """
    Takes the system and feeds in 1101 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([1, 1, 0, 1]))


def fixed_input_binary_14(system: System):
    """
    Takes the system and feeds in 1110 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([1, 1, 1, 0]))


def fixed_input_binary_15(system: System):
    """
    Takes the system and feeds in 1111 the system and returns the float converted output.

    Parameters
    -----------
    system  :   System
        PyCharc system

    Return
    -------
    float
        base 10 expression of binary output from system.
    """
    return fixed_input_binary(system, np.array([1, 1, 1, 1]))
