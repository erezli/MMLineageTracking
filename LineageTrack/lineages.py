from LineageTrack.cells import Cell

from scipy.stats import linregress
from math import isnan
import numpy as np


class Lineage:
    def __init__(self, cells):
        self.trench = cells[0].trench
        self.resident_time = [cell.time for cell in cells]
        self.labels = [cell.label for cell in cells]
        self.lengths = [cell.major for cell in cells]
        self.widths = [cell.minor for cell in cells]
        self.positions = [(cell.centroid_x, cell.centroid_y) for cell in cells]
        self.barcode = cells[0].barcode
        self.pole_label = cells[0].poles

        self.daughters = (None, None)
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def set_daughters(self, daughters):
        if len(daughters) == 1:
            if self.daughters[0] is None:
                self.daughters = (daughters[0], None)
            elif self.daughters[1] is None:
                self.daughters = (self.daughters[0], daughters[0])
            else:
                self.daughters = (daughters[0], None)
        elif len(daughters) == 2:
            self.daughters = (daughters[0], daughters[1])

    def evolution(self):
        return [(self.resident_time[i], self.labels[i], self.lengths[i], self.widths[i],
                 self.positions[i], self.barcode[i], self.pole_label[i])
                for i in range(len(self.resident_time))]

    def __iter__(self):
        return iter(self.evolution())

    @classmethod
    def from_tracker(cls, cells_list):
        lineages = []
        mother_cell = cells_list[0][0]
        next_line = [(mother_cell, None)]

        def track_line(cell1, parent):
            line = [cell1]
            while cell1.divide is False and cell1.daughters:
                cell1 = cell1.daughters
                line.append(cell1)
            lineages.append(cls(line))
            lineages[-1].set_parent(parent)
            if parent is not None:
                parent.set_daughters([lineages[-1]])
            return cell1

        while len(next_line) != 0:
            end_cell = track_line(next_line[0][0], next_line[0][1])
            next_line.pop(0)
            if isinstance(end_cell.daughters, tuple):
                next_line.extend([(d, lineages[-1]) for d in end_cell.daughters])
            elif isinstance(end_cell.daughters, Cell):
                next_line.append((end_cell.daughters, lineages[-1]))

        return lineages

    def get_adder_dl(self):
        if self.daughters[0] is not None:
            return self.lengths[-1] - self.lengths[0]
        else:
            return None

    def get_timer_dt(self):
        if self.daughters[0] is not None:
            return self.daughters[0].resident_time[0] - self.resident_time[0]
        else:
            return None

    def get_growth_time_constant(self):
        if self.daughters[0] is not None:
            slope, inter, r, p, se = linregress(self.resident_time - self.resident_time[0], np.log2(self.lengths))
            if not isnan(slope):
                return slope
            else:
                return None
        else:
            return None
