import os
from glob import glob
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress


def reconstruct_array_from_str(str_version_of_array):
    """
    Reconstructs the string version of a numpy array which gets saved to csv format by pandas.
    """
    unformatted_pixel_lists = str_version_of_array.split("\n")
    formatted_pixel_lists = [re.split(r'[\]\[\s+]\s*', unformatted_pixel_lists[x]) for x in
                             range(len(unformatted_pixel_lists))]
    null_data_counter = 0
    for count, row in enumerate(formatted_pixel_lists):
        # new_row = [int(x) for x in row if x != ""]
        new_row = []
        for x in row:
            if x != "":
                try:
                    new_row.append(int(x))
                except ValueError:
                    null_data_counter += 1
                    # print(row)
                    new_row.append(0)
        formatted_pixel_lists[count] = new_row
    if null_data_counter != 0:
        # print(null_data_counter)
        print("Some data is lost due to compression in image intensity column")
    pixel_array = np.array(formatted_pixel_lists)

    return pixel_array


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
        self.local_centroid_y = properties["centroid_local-0"]
        self.local_centroid_x = properties["centroid_local-1"]
        # self.orientation = properties["orientation"]
        self.channel_intensities = []
        for c in channels:
            self.channel_intensities.append(properties["{}_intensity_mean".format(c)])
        if reporter is not None:
            self.reporter_intensities = properties["{}_intensity_mean".format(reporter)]

    def __str__(self):
        return f"""cell in trench {self.trench} at {self.time} min with label {self.label}"""

    def get_point_coordinate(self, mode="no_division"):
        if mode == "no_division":
            coordinate = [self.area, self.major, self.minor, self.centroid_x, self.centroid_y, self.local_centroid_x,
                          self.local_centroid_y]   # , self.orientation]
            for i in self.channel_intensities:
                coordinate.append(i)
            return np.array(coordinate)
        elif mode == "upper_division":
            coordinate = [self.area * 2, self.major * 2, self.minor, self.centroid_x,
                          self.centroid_y + self.local_centroid_y, self.local_centroid_x,
                          self.local_centroid_y * 2]   # , self.orientation]
            for i in self.channel_intensities:
                coordinate.append(i)
            return np.array(coordinate)
        elif mode == "lower_division":
            coordinate = [self.area * 2, self.major * 2, self.minor, self.centroid_x,
                          self.centroid_y - self.local_centroid_y, self.local_centroid_x,
                          self.local_centroid_y * 2]    # , self.orientation]
            for i in self.channel_intensities:
                coordinate.append(i)
            return np.array(coordinate)
        else:
            raise Exception("specified mode unknown")


class LineageTrack:
    def __init__(self, filepath, *args):
        if os.path.isdir(filepath):
            self.directory = filepath
            self.files = glob(self.directory + "{}*".format(os.path.sep))
        elif os.path.exists(filepath):
            self.files = [filepath]
        else:
            raise Exception("specified file/directory not found")
        for path in args:
            if not os.path.exists(path):
                print("invalid argument(s) - should be paths to the files or a single directory")
            else:
                self.files.append(path)
        print("Looking for data at these locations:")
        for f in self.files:
            print(f)

        # df_list = [pd.read_csv(f, converters = {"image_intensity":reconstruct_array_from_str}) for f in self.files]
        cols = list(pd.read_csv(self.files[0], nrows=1))
        columnes_to_skip = list(["image_intensity", "orientation"])
        df_list = [pd.read_csv(f, usecols=[i for i in cols if i not in columnes_to_skip],
                               dtype={"major_axis_length": np.float32, "minor_axis_length": np.float32,
                                      "centroid-0": np.float32, "centroid-1": np.float32,
                                      "centroid_local-0": np.float32, "centroid_local-1": np.float32,
                                      "orientation": np.float32, "intensity_mean": np.float32})
                                      #"trench_id": np.uint8, "time_(mins)": np.uint8, "label": np.uint8})
                   # converters={"image_intensity": reconstruct_array_from_str})
                   for f in self.files]
        # Todo: use Zarr array to reduce memory usage
        self.channels = []
        # list_of_properties = ["trench_id", "time_(mins)", "label", "area", "major_axis_length",
        #                      "minor_axis_length", "centroid-0", "centroid-1",
        #                      "centroid_local-0", "centroid_local-1", "orientation"]
        list_of_properties = ["trench_id", "time_(mins)", "label", "area", "major_axis_length",
                              "minor_axis_length", "centroid-0", "centroid-1",
                              "centroid_local-0", "centroid_local-1"]
        self.df = df_list[0][list_of_properties].copy()
        for d in df_list:
            channel = d.loc[1, "channel"]
            self.channels.append(channel)
            # self.df.loc[:, "{}_intensity_mean".format(channel)] = d.loc[:, "intensity_mean"]
            self.df.insert(self.df.shape[1], "{}_intensity_mean".format(channel), d.loc[:, "intensity_mean"])
            # might include later but takes a lot of memory
            # self.df.insert(len(self.df.columns),"{}_image_intensity".format(channel), d.loc[:, "image_intensity"])
            # self.df.insert(len(self.df.columns), "{}_identity".format(channel), d.loc[:, "identity"])
        self.channels = sorted(list(set(self.channels)))
        self.properties = sorted(list(set(self.df.columns)))
        self.df.sort_values(["trench_id", "time_(mins)", "label"], inplace=True)
        self.trenches = self.df.loc[:, "trench_id"]
        self.trenches = sorted(list(set(self.trenches)))
        self.current_trench = 0
        self.current_frame = 0
        self.next_frame = 0
        self.frames = []
        self.buffer = None
        self.current_cell_number = 0
        self.next_track = []
        self.current_lysis = []
        self.div_intervals = []
        self.div_interval = 24  # theoretical time for division
        self.growth_taus = []
        self.growth_tau = 0.03


        print("Finished loading the data")
        print(self.df.head(1))
        print(self.df.shape)

    def __str__(self):
        return f"""
            Read {len(self.files)} files
            Channels: {self.channels}
            Properties for each cell: {self.properties}
        """

    def get_mother_cell_growth(self, trench, plot=False):
        # user defines by examining the data the period during which cells grow exponentially
        if type(trench) is list:
            subplots = len(trench)
            cols = 2
            rows = round(np.ceil(subplots / cols))
            fig, axes = plt.subplots(nrows=rows, ncols=cols, dpi=80, figsize=(20, 20))
            axes_flat = axes.flatten()
            mcells = []
            for i, tr in enumerate(trench):
                if tr in self.trenches:
                    mother_cell = self.df.loc[(self.df["label"] == 1) & (self.df["trench_id"] == tr),
                                              ["time_(mins)", "major_axis_length"]].copy()
                    mother_cell = mother_cell.to_numpy()
                    mcells.append(mother_cell)
                    if plot:
                        axes_flat[i].plot(mother_cell[:, 0], mother_cell[:, 1])
                        axes_flat[i].set_title(f"trench_ids: {tr}, the major axis length of the mother cell")
            # fig.show()
            return mcells
        else:
            if trench in self.trenches:
                mother_cell = self.df.loc[(self.df["label"] == 1) & (self.df["trench_id"] == trench),
                                          ["time_(mins)", "major_axis_length"]].copy()
                mother_cell = mother_cell.to_numpy()
                if plot:
                    plt.figure(figsize=(10, 10))
                    plt.plot(mother_cell[:, 0], mother_cell[:, 1])
                    plt.title(f"trench_id: {trench}, the major axis length of the mother cell")
                    plt.show()
                return mother_cell

    def find_division(self, trench):
        mcell = self.get_mother_cell_growth(trench)
        idx_p = find_peaks(mcell[:, 1], threshold=1, distance=3)
        peaks = [mcell[p, :] for p in idx_p[0]]
        peaks = np.array(peaks)
        plt.figure(figsize=(10, 10))
        plt.plot(mcell[:, 0], mcell[:, 1])
        plt.plot(peaks[:, 0], peaks[:, 1], "x")
        plt.title(f"trench_id: {trench}, the major axis length of the mother cell")
        plt.show()
        return mcell, idx_p

    def collect_model_para(self, mcell, e_phase_idx, plot=False):
        """
        estimate parameters for the model of cells growth and division in exponential phase
        *THIS SHOULD ONLY RUN ONCE FOR EACH MOTHER CELL IN EVERY TRENCH*
        :param e_phase_idx: sliced index 1D array for exponential phase
        :return: average division time interval
        time constant - tau, for the exponential growth model
        """
        e_peaks = [mcell[p, :] for p in e_phase_idx]
        e_peaks = np.array(e_peaks)

        division_times = []
        growth_taus = []

        intervals = [e_peaks[i + 1][0] - e_peaks[i][0] for i in range(len(e_peaks) - 1)]
        division_times += intervals
        print(division_times)

        for i in range(len(e_phase_idx) - 1):
            growth = mcell[e_phase_idx[i] + 1:e_phase_idx[i + 1]]
            slope, inter, r, p, se = linregress(growth[:, 0] - growth[0, 0], np.log2(growth[:, 1]))
            if plot:
                plt.plot(growth[:, 0] - growth[0, 0], np.log2(growth[:, 1]))
                x = np.linspace(0, 20, 100)
                plt.plot(x, slope*x + inter)
                print(f"the slope is estimated to be {slope}")
                print(f"the intercept is estimated to be {inter}")
                plt.show()
            growth_taus.append(slope)

        self.div_intervals += division_times
        self.growth_taus += growth_taus

    def update_model_para(self):
        self.div_interval = np.mean(self.div_intervals)
        self.growth_tau = np.mean(self.growth_taus)
        print(f"""
                The average time interval for division is {self.div_interval}
                The time constant for exponential growth is {self.growth_tau}""")

    def load_trench_current_frame(self, channels):
        if self.buffer is not None:
            return self.buffer[0], self.buffer[1]
        else:
            current_local_data = self.df.loc[(self.df["trench_id"] == self.current_trench)
                                             & (self.df["time_(mins)"] == self.current_frame)].copy()
            ###
            #columns = [col for col in self.properties if col not in ["trench_id", "time_(mins)", "label"]]
            #for c in columns:
            #    current_local_data[c] = MinMaxScaler().fit_transform(np.array(current_local_data[c]).reshape(-1, 1))
            ###
            cells_list = [Cell(row, channels) for index, row in current_local_data.iterrows()]
            points = [cell.get_point_coordinate("no_division") for cell in cells_list]
            return cells_list, np.array(points)

    def load_trench_next_frame(self, channels):
        next_local_data = self.df.loc[(self.df["trench_id"] == self.current_trench)
                                      & (self.df["time_(mins)"] == self.next_frame)].copy()
        ###
        #columns = [col for col in self.properties if col not in ["trench_id", "time_(mins)", "label"]]
        #for c in columns:
        #    next_local_data[c] = MinMaxScaler().fit_transform(np.array(next_local_data[c]).reshape(-1, 1))
        ###
        cells_list = [Cell(row, channels) for index, row in next_local_data.iterrows()]
        points_1 = [cell.get_point_coordinate("no_division") for cell in cells_list]
        points_2 = [cell.get_point_coordinate("upper_division") for cell in cells_list]
        points_3 = [cell.get_point_coordinate("lower_division") for cell in cells_list]
        self.buffer = (cells_list, np.array(points_1))
        return cells_list, [np.array(points_1), np.array(points_2), np.array(points_3)]

    def compare_distance(self, distances, idx1, idx2, idx3):
        pointers = distances.idxmin(axis=1)
        nn_list = []
        for n in range(pointers.shape[0]):
            if pointers[n] == "d1":
                nn_list.append(idx1[n])
            elif pointers[n] == "d2":
                # upper division - need to label
                print("division 1!")
                nn_list.append(idx2[n])
            elif pointers[n] == "d3":
                # lower division - need to label
                print("division 2!")
                nn_list.append(idx3[n])
            else:
                raise ValueError
        self.next_track = nn_list

    def triangulation(self, current_points, predictions):
        # current_points = normalize(current_points, axis=0)
        # current_points[:, 4] = current_points[:, 4] * 20     # add weighting to the centroid_y
        # current_points[:, 0] = current_points[:, 0] * 5      # add weighting to the area
        print("the points coordinate of current trench: ")
        print(current_points)
        grid = KDTree(current_points)
        for i in range(len(predictions)):
            # predictions[i] = normalize(predictions[i], axis=0)
            # predictions[i][:, 4] = predictions[i][:, 4] * 20  # add weighting to the centroid_y
            # predictions[i][:, 0] = predictions[i][:, 0] * 5  # add weighting to the area

            print(predictions[i])
        distance1, idx1 = grid.query(predictions[0], workers=-1)
        distance2, idx2 = grid.query(predictions[1], workers=-1)
        distance3, idx3 = grid.query(predictions[2], workers=-1)
        df = pd.DataFrame(data={
            "d1": distance1,
            "d2": distance2,
            "d3": distance3
        })
        self.compare_distance(df, idx1, idx2, idx3)

    def lysis_cells(self):
        idx_list = range(self.current_cell_number)
        self.current_lysis = [i for i in idx_list if i not in self.next_track]

    def track_cells(self, special_reporter=None):
        for tr in self.trenches:
            tr += 1 # for testing - look at different trench
            self.current_trench = tr
            self.frames = self.df.loc[self.df["trench_id"] == self.current_trench, "time_(mins)"].copy()
            self.frames = sorted(list(set(self.frames)))
            for i in range(len(self.frames) - 1):
                self.current_frame = self.frames[i]
                self.next_frame = self.frames[i + 1]
                if special_reporter is None:
                    current_cells, grid = self.load_trench_current_frame(self.channels)
                    next_cells, points = self.load_trench_next_frame(self.channels)
                    self.current_cell_number = len(current_cells)

                    print("looking at cells: ")
                    for x in range(len(current_cells)):
                        print(current_cells[x])
                    # print(grid)

                    self.triangulation(grid, points)
                    # index pointers to the current frame cells from next frame
                    self.lysis_cells()
                    # index of the current frame cells
                    list_track = [i+1 for i in self.next_track ]
                    print(list_track)
                    list_lyse = [i+1 for i in self.current_lysis]
                    print(list_lyse)
                else:
                    if special_reporter in self.channels:
                        """
                        take the reporter channel out of the channel list, pass to the cell object, 
                        so its data will not be included into the coordinates
                        """
                    else:
                        raise Exception("the specified channel has no data found")
                ### for testing ###
                if i == 0:
                    break
            self.buffer = None
            ### for testing ###
            break
