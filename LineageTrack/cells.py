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
        self.parent_label = None
        self.parent = None
        self.daughters = None
        self.barcode = None

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

    def set_daughters(self, daughter):
        if self.daughters is None:
            self.daughters = daughter
        else:
            self.daughters = (self.daughters, daughter)

    def assign_barcode(self, to_whom="daughter", second=False, max_bit=8):
        if to_whom == "daughter":
            if (self.barcode is not None) and (self.daughters is not None):
                if len(self.barcode) < max_bit:
                    if isinstance(self.daughters, tuple):
                        self.daughters[0].barcode = self.barcode + [0]
                        self.daughters[1].barcode = self.barcode + [1]
                    else:
                        self.daughters.barcode = self.barcode
        elif to_whom == "self":
            if (self.parent is not None) and (self.parent.barcode is not None):
                if len(self.parent.barcode) < max_bit:
                    if not self.parent.divide:
                        self.barcode = self.parent.barcode
                    elif second:
                        self.barcode = self.parent.barcode + [1]
                    else:
                        self.barcode = self.parent.barcode + [0]

    def barcode_to_binary(self, max_bit=8):
        if self.barcode:
            while len(self.barcode) < max_bit:
                self.barcode.append(0)
            self.barcode = bin(int("".join(map(str, self.barcode)), 2))
