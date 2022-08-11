import numpy as np


class Cell:
    def __init__(self, properties, channels, reporter=None):
        self.trench = properties["trench_id"]
        self.time = properties["time_(mins)"]
        self.label = properties["label"]
        # self.area = properties["area"]
        self.major = properties["major_axis_length"]
        self.minor = properties["minor_axis_length"]
        self.centroid_y = properties["centroid-0"]
        self.centroid_x = properties["centroid-1"]
        # self.local_centroid_y = properties["centroid_local-0"]
        # self.local_centroid_x = properties["centroid_local-1"]
        # self.orientation = properties["orientation"]
        self.channel_intensities = []
        for c in channels:
            self.channel_intensities.append(properties["{}_intensity_mean".format(c)])
        if reporter is not None:
            self.reporter_intensities = properties["{}_intensity_mean".format(reporter)]
        self.coord = []
        self.divide = False
        self.parent = None
        self.bin_label = None

    def __str__(self):
        return f"""cell in trench {self.trench} at {self.time} min with label {self.label}"""

    def set_coordinates(self, division=0, growth=1, offset=0, reset_original=False):
        if reset_original:
            self.coord = np.array([[self.major, self.centroid_y]])
            return self.coord
        if division == 0:
            # self.coord = [self.area, self.major, self.minor, self.centroid_x, self.centroid_y, self.local_centroid_x,
            #              self.local_centroid_y, self.orientation]
            self.coord = np.array([[self.major * growth, self.centroid_y + offset + self.major * (growth - 1) / 2]])
            # for i in self.channel_intensities:
            #     self.coord.append(i)
            self.divide = False
            return self.coord
        elif division == 1:
            # self.coord = [self.area * 2, self.major * 2, self.minor, self.centroid_x,
            #               self.centroid_y + self.local_centroid_y, self.local_centroid_x,
            #               self.local_centroid_y * 2, self.orientation]
            self.coord = np.array(
                [[self.major * growth / 2, self.centroid_y + offset + self.major * (2 * growth - 3) / 4],
                 [self.major * growth / 2, self.centroid_y + offset + self.major * (2 * growth - 1) / 4]])
            # for i in self.channel_intensities:
            #     self.coord.append(i)
            self.divide = True
            return self.coord
        else:
            raise Exception("specified division unknown")

    def set_parent(self, parent_cell):
        self.parent = parent_cell

    def assign_label(self, second=False):
        if self.parent is not None:
            if self.parent.bin_label is not None:
                if not self.parent.divide:
                    self.bin_label = self.parent.bin_label
                elif second:
                    self.bin_label = self.parent.bin_label.append(1)
                else:
                    self.bin_label = self.parent.bin_label.append(0)
