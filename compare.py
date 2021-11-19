#!/usr/bin/env python3

from __future__ import print_function

import numpy as np
import sys

from util import io

usage = """
      usage: ./compare.py file1 file2 (rtol)

      where rtol is an (optional) relative tolerance parameter to use when
      comparing the data
"""

errors = {"gridbad": "grids don't agree",
          "namesbad": "variable lists don't agree",
          "varerr": "one or more variables don't agree"}


def compare(data1, data2, rtol=1.e-12):
    """
    given two CellCenterData2d objects, compare the data, zone-by-zone
    and output any errors

    Parameters
    ----------
    data1, data2 : CellCenterData2d object
        Two data grids to compare
    rtol : float
        relative tolerance to use to compare grids

    """

    # compare the grids
    if not data1.grid == data2.grid:
        return "gridbad"

    # compare the data
    if not sorted(data1.names) == sorted(data2.names):
        return "namesbad"

    print(" ")
    print("variable comparisons:")

    result = 0

    for name in data1.names:

        d1 = data1.get_var(name)
        d2 = data2.get_var(name)

        abs_err = np.max(np.abs(d1.v() - d2.v()))

        if not np.any(d2.v() == 0):
            rel_err = np.max(np.abs(d1.v() - d2.v()) / np.abs(d2.v()))
            print("{:20s} absolute error = {:10.10g}, relative error = {:10.10g}".format(name, abs_err, rel_err))
        else:
            print("{:20s} absolute error = {:10.10g}".format(name, abs_err))

        if not np.allclose(d1.v(), d2.v(), rtol=rtol):
            result = "varerr"

    return result


if __name__ == "__main__":

    if not (len(sys.argv) == 3 or len(sys.argv) == 4):
        print(usage)
        sys.exit(2)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    s1 = io.read(file1)
    s2 = io.read(file2)

    if len(sys.argv) == 3:
        result = compare(s1.cc_data, s2.cc_data)
    else:
        result = compare(s1.cc_data, s2.cc_data, rtol=float(sys.argv[3]))

    if result == 0:
        print("SUCCESS: files agree")
    else:
        print("ERROR: ", errors[result])
