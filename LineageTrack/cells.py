import numpy as np
import copy
import ast


class Cell:
    def __init__(self, properties, channels, reporter=None):
        self.trench = properties["trench_id"]
        self.time = properties["time_(mins)"]
        self.label = properties["label"]
        self.area = properties["area"]
        self.major = properties["major_axis_length"]
        self.minor = properties["minor_axis_length"]
        self.centroid_y = properties["centroid-0"]
        self.centroid_x = properties["centroid-1"]
        # self.local_centroid_y = properties["centroid_local-0"]
        # self.local_centroid_x = properties["centroid_local-1"]
        self.orientation = properties["orientation"]
        # useful moments: almost all
        self.zernike = ast.literal_eval(properties["zernike"])
        self.zernike_half = ast.literal_eval(properties["zernike_half"])
        self.channel_intensities = {}
        for c in channels:
            self.channel_intensities[c] = properties["{}_intensity_mean".format(c)]
        if reporter is not None:
            self.reporter_intensity = properties["{}_intensity_mean".format(reporter)]
            self.reporter_total_intensity = properties["{}_intensity_total".format(reporter)]
        else:
            self.reporter_intensity = None
        self.coord = []
        self.divide = False
        self.out = False
        self.lyse = False
        self.parent_label = None
        self.parent = None
        self.daughters = None
        self.barcode = None
        self.poles = None
        self.within_safe_zone = False

    def __str__(self):
        return f"""cell in trench {self.trench} at {self.time} min with label {self.label}"""

    def set_coordinates(self, division=0, growth=1, offset=0, reset_original=False, radius=0):
        """
        update the cell's major axis length and y position
        @param division: 0 is no division; 1 means there is division
        @param growth: growth rate, no growth if equals to one
        @param offset: offsets caused by other cells growing
        @param reset_original: return the cell's original length and position if set to True
        @return: a 2D array of one or two [length, y position]
        """
        if reset_original:
            self.coord = np.array([[self.major, self.centroid_y, np.sqrt(self.area), np.sum(np.abs(self.zernike))]])
            return self.coord
        if division == 0:
            # self.coord = [self.area, self.major, self.minor, self.centroid_x, self.centroid_y, self.local_centroid_x,
            #              self.local_centroid_y, self.orientation]
            self.coord = np.array([[self.major * growth,
                                    self.centroid_y + offset + self.major * (growth - 1) / 2,
                                    np.sqrt(self.area) * growth, 
                                    np.sum(np.abs(self.zernike)) * radius]]) # radius of the zernike
            # for i in self.channel_intensities:
            #     self.coord.append(i)
            self.divide = False
            return self.coord
        elif division == 1:
            # self.coord = [self.area * 2, self.major * 2, self.minor, self.centroid_x,
            #               self.centroid_y + self.local_centroid_y, self.local_centroid_x,
            #               self.local_centroid_y * 2, self.orientation]
            self.coord = np.array(
                [[self.major * growth / 2 * 0.9,  # 0.9 is segmentation erosion
                  self.centroid_y + offset + self.major * (growth - 2) / 4,
                  np.sqrt(self.area) * growth / 2 * 0.9, 
                  np.sum(np.abs(self.zernike_half[0])) * radius],
                 [self.major * growth / 2 * 0.9,  # 0.9 is segmentation erosion
                  self.centroid_y + offset + self.major * (3 * growth - 2) / 4,
                  np.sqrt(self.area) * growth / 2 * 0.9, 
                  np.sum(np.abs(self.zernike_half[1])) * radius]])
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
                    if self.divide:
                        if isinstance(self.daughters, tuple):
                            self.daughters[0].barcode = self.barcode + [0]
                            self.daughters[1].barcode = self.barcode + [1]
                        else:
                            self.daughters.barcode = self.barcode + [0]
                    else:
                        self.daughters.barcode = copy.deepcopy(self.barcode)
        elif to_whom == "self":
            if (self.parent is not None) and (self.parent.barcode is not None):
                if len(self.parent.barcode) < max_bit:
                    if not self.parent.divide:
                        self.barcode = copy.deepcopy(self.parent.barcode)
                    elif second:
                        self.barcode = self.parent.barcode + [1]
                    else:
                        self.barcode = self.parent.barcode + [0]

    def barcode_to_binary(self, max_bit=8):
        if isinstance(self.barcode, list):
            while len(self.barcode) < max_bit:
                self.barcode.append(0)
            self.barcode = bin(int("".join(map(str, self.barcode)), 2))

    def set_generation(self):
        if (self.poles is not None) and (self.daughters is not None):
            if self.divide:
                if isinstance(self.daughters, tuple):
                    self.daughters[0].poles = (self.poles[0] + 1, 0)
                    self.daughters[1].poles = (0, self.poles[1] + 1)
                else:
                    self.daughters.poles = (self.poles[0] + 1, 0)
            else:
                self.daughters.poles = copy.deepcopy(self.poles)
