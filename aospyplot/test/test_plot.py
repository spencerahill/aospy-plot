#!/usr/bin/env python
"""Unit tests of aospy-plot Plot class and associated functionality."""

import sys
import unittest


class PlotTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestPlot(PlotTestCase):
    def test_import(self):
        from aospyplot import plot, Plot, PlotInterface


if __name__ == '__main__':
    sys.exit(unittest.main())
