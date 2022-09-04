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
        self.areas = [cell.area for cell in cells]
        self.reporter_intensities = [cell.reporter_intensity for cell in cells]
        self.barcode = cells[0].barcode
        self.pole_label = cells[0].poles

        if cells[-1].lyse:
            self.lyse = True
        else:
            self.lyse = False
        self.daughters = (None, None)
        self.parent = None

        self.highlight = False

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
        for mother_cell in cells_list:
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
        if self.daughters[0] is not None and self.parent is not None:
            if self.lengths[-1] > self.lengths[0]:  # Todo: better condition for outliers?
                return self.lengths[-1] - self.lengths[0]
            else:
                return None
        else:
            return None

    def get_timer_dt(self):
        if self.daughters[0] is not None and self.parent is not None:
            return self.daughters[0].resident_time[0] - self.resident_time[0]
        else:
            return None

    def get_growth_time_constant(self):
        if self.daughters[0] is not None and self.parent is not None and len(self.resident_time) > 2:
            slope, inter, r, p, se = linregress([t - self.resident_time[0] for t in self.resident_time],
                                                np.log2(self.lengths))
            if not isnan(slope):
                return 1 / slope
            else:
                print(self.resident_time)
                print(self.labels)
                return None
        else:
            return None

    def instant_growth_rate_position(self):
        for i in range(len(self.resident_time) - 1):
            dlogL = np.log2(self.lengths[i + 1]) - np.log2(self.lengths[i])
            dt = self.resident_time[i + 1] - self.resident_time[i]
            yield dlogL / dt, self.positions[i][1]

    def partial_dlogA_dt(self):
        for i in range(len(self.resident_time) - 1):
            dlogA = np.log2(self.areas[i + 1]) - np.log2(self.areas[i])
            dt = self.resident_time[i + 1] - self.resident_time[i]
            if not isnan(dlogA / dt):
                yield dlogA / dt, (self.resident_time[i], self.labels[i])
