import os
from glob import glob
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.spatial import KDTree, distance_matrix
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress, poisson
import itertools
import copy
from math import comb
from LineageTrack.cells import Cell
from scipy.stats import norm
from tqdm import tqdm


def reconstruct_array_from_str(str_version_of_array):
    """
    Reconstructs the string version of a numpy array which gets saved to csv format by pandas. - Charlies
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


def nearest_neighbour(points, true_coord, mode="KDTree"):
    if mode == "KDTree":
        grid = KDTree(points)
        # idx points to the index of points constructing the KDTree
        distance, idx = grid.query(true_coord, workers=-1)
        return distance, idx
    if mode == "SeqMatch":
        # match exclusively
        d_mtx = distance_matrix(true_coord, points)
        distance = []
        idx = []
        index_mtx = np.argsort(d_mtx, axis=1)
        for row in range(index_mtx.shape[0]):
            for col in range(index_mtx.shape[1]):
                if index_mtx[row][col] not in idx:
                    idx.append(index_mtx[row][col])
                    distance.append(d_mtx[row][index_mtx[row][col]])
                    break
                elif col == index_mtx.shape[1] - 1:
                    # for the case next frame has more cells than the simulated scenario
                    idx.append(None)
                    distance.append(d_mtx[row][col])
        return distance, idx


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

        ### load data ###
        # df_list = [pd.read_csv(f, converters = {"image_intensity":reconstruct_array_from_str}) for f in self.files]
        cols = list(pd.read_csv(self.files[0], nrows=1))
        columnes_to_skip = list(["image_intensity", "orientation", "centroid_local-0", "centroid_local-1"])
        df_list = [pd.read_csv(f, usecols=[i for i in cols if i not in columnes_to_skip],
                               dtype={"major_axis_length": np.float32, "minor_axis_length": np.float32,
                                      "centroid-0": np.float32, "centroid-1": np.float32,
                                      "orientation": np.float32, "intensity_mean": np.float32})
                   # "trench_id": np.uint8, "time_(mins)": np.uint8, "label": np.uint8,
                   # "centroid_local-0": np.float32, "centroid_local-1": np.float32}
                   # converters={"image_intensity": reconstruct_array_from_str})
                   for f in self.files]
        # Todo: use Zarr array to reduce memory usage
        self.channels = []
        # list_of_properties = ["trench_id", "time_(mins)", "label", "area", "major_axis_length",
        #                      "minor_axis_length", "centroid-0", "centroid-1",
        #                      "centroid_local-0", "centroid_local-1", "orientation"]
        list_of_properties = ["trench_id", "time_(mins)", "label", "area", "major_axis_length",
                              "minor_axis_length", "centroid-0", "centroid-1"]
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
        self.max_y = self.df.loc[:, "centroid-0"].max()

        ### for tracking ###
        self.current_trench = 0
        self.current_frame = 0
        self.next_frame = 0
        self.dt = 0
        self.current_number_of_cells = 0
        self.buffer = None

        ### model parameters ###
        self.div_intervals = []
        self.div_interval = 24  # theoretical time for division
        self.growth_taus = []
        self.growth_tau = 24
        self.length_at_div = []
        self.div_length_paras = (50, 20)
        self.mother_cell_collected = []

        self.next_track = []
        self.current_lysis = []

        print("Finished loading the data")
        # print(f"Head: {self.df.head(1)}")
        print(self.df.shape)

    def __str__(self):
        return f"""
            Read {len(self.files)} files
            Channels: {self.channels}
            Properties for each cell: {self.properties}
        """

    def get_mother_cell_growth(self, trench, plot=False):
        """
        Extract mother cell information from given trench
        @param trench: the trench id(s) for the mother cell you want to examine
        @type trench: int or list
        @param plot: plot the length of the mother cell(s) against time - for debug purpose
        @return: a numpy array containing the length and time of the mother cell
        """
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
        """
        find and label the division by searching for peaks in the length-time plot, may need user to slice out the bad
        results
        @param trench: the trench id
        @return: first entry is a tuple of the trench id and the length-time array of the mother cell, the second entry
        is a 1D array for index of the peaks
        """
        mcell = self.get_mother_cell_growth(trench)
        idx_p = find_peaks(mcell[:, 1], threshold=1, distance=3)
        peaks = [mcell[p, :] for p in idx_p[0]]
        peaks = np.array(peaks)
        plt.figure(figsize=(10, 10))
        plt.plot(mcell[:, 0], mcell[:, 1])
        plt.plot(peaks[:, 0], peaks[:, 1], "o")
        plt.title(f"trench_id: {trench}, the major axis length of the mother cell")
        plt.show()
        mother_cell_info = (trench, mcell)
        return mother_cell_info, idx_p[0]

    def collect_model_para(self, mother_cell, e_phase_idx, plot=False):
        """
        estimate parameters for the model of cells growth and division in exponential phase
        user defines the period during which cells grow exponentially by examining the data
        *THIS SHOULD ONLY RUN ONCE FOR EACH MOTHER CELL IN EVERY TRENCH*
        @type mother_cell: tuple
        @param mother_cell: first entry is the trench id, second entry is the length and time of the mother cell
        @param e_phase_idx: sliced index 1D array for exponential phase
        This collects division time interval, time constant - tau, for the exponential growth model,
        and the max length of cells
        """
        if mother_cell[0] in self.mother_cell_collected:
            return f"""this mother cell at trench {mother_cell[0]} has already been collected"""
        else:
            self.mother_cell_collected.append(mother_cell[0])
            e_peaks = [mother_cell[1][p, :] for p in e_phase_idx]
            e_peaks = np.array(e_peaks)

            growth_taus = []

            division_intervals = [e_peaks[i + 1][0] - e_peaks[i][0] for i in range(len(e_peaks) - 1)]
            division_times = list(e_peaks[:, 0])
            max_length = list(e_peaks[:, 1])

            for i in range(len(e_phase_idx) - 1):
                growth = mother_cell[1][e_phase_idx[i] + 1:e_phase_idx[i + 1]]
                slope, inter, r, p, se = linregress(growth[:, 0] - growth[0, 0], np.log2(growth[:, 1]))
                if plot:
                    plt.plot(growth[:, 0] - growth[0, 0], np.log2(growth[:, 1]))
                    x = np.linspace(0, 24, 100)
                    plt.plot(x, slope * x + inter)
                    print(f"the slope is estimated to be {slope}")
                    print(f"the intercept is estimated to be {inter}")
                    plt.show()
                growth_taus.append(slope)

            self.div_intervals += division_intervals
            self.growth_taus += growth_taus
            self.length_at_div.append([division_times, max_length])

    def update_model_para(self, model="unif"):
        # Todo: for cells growing heterogeneously or the growth distribution changes over time - collect model
        #  parameter across mother cells
        if model == "unif":
            self.div_interval = np.mean(self.div_intervals)
            self.growth_tau = np.mean(self.growth_taus)
            self.growth_tau = 1 / self.growth_tau
            max_lengths = []
            for i in self.length_at_div:
                max_lengths += i[1]
            mean = np.mean(max_lengths)
            var = np.var(max_lengths)
            self.div_length_paras = (mean, var)
            print(f"""
                    The average time interval for division is {self.div_interval}
                    The time constant for exponential growth is {self.growth_tau}
                    The average division length is {self.div_length_paras[0]} with variance {self.div_length_paras[1]}
                    """)

    def calculate_growth(self):
        return 2 ** (self.dt / self.growth_tau)

    def pr_div_lambda(self, n):
        # n = 0: no division
        # n = 1: division
        return poisson.pmf(n, self.dt / self.growth_tau)

    def pr_div_length(self, n, length):
        pr_d = norm.cdf(length, loc=self.div_length_paras[0], scale=np.sqrt(self.div_length_paras[1])) * n
        pr_no_d = (1 - norm.cdf(length, loc=self.div_length_paras[0], scale=np.sqrt(self.div_length_paras[1]))) * (1 - n)
        return pr_d + pr_no_d

    def cells_simulator(self, cells_list, max_dpf, p_sp):
        growth = self.calculate_growth()
        cells_futures = []
        cells_state = []
        prob_0 = self.pr_div_lambda(0)
        prob_1 = self.pr_div_lambda(1)
        if self.buffer is not None:
            cells_futures.append([p_sp, copy.deepcopy(self.buffer)])
            cells_state.append(["SP"] * len(self.buffer))
        else:
            for i in range(len(cells_list)):
                cells_list[i].set_coordinates(0)
            cells_futures.append([p_sp, copy.deepcopy(cells_list)])
            cells_state.append(["SP"] * len(cells_list))
        for d in range(max_dpf + 1):
            d_list = []
            d_list.extend([1] * d)
            d_list.extend([0] * (self.current_number_of_cells - d))
            combinations = set(list(itertools.permutations(d_list)))
            for com in combinations:
                offset = 0
                prob = 1
                for i in range(self.current_number_of_cells):
                    cells_list[i].set_coordinates(division=com[i], growth=growth, offset=offset)
                    if com[i] == 0:
                        offset += cells_list[i].coord[0][0] * (1 - 1 / growth)
                        prob *= prob_0 * self.pr_div_length(0, cells_list[i].coord[0][0])
                    elif com[i] == 1:
                        offset += cells_list[i].coord[0][0] * (1 - 1 / growth) * 2
                        prob *= prob_1 * self.pr_div_length(1, cells_list[i].coord[0][0]*2)
                    else:
                        print("There's something wrong with the list of combination of division and no division:")
                        print(com)
                cells_futures.append([prob * comb(self.current_number_of_cells, d), copy.deepcopy(cells_list)])
                cells_state.append(["Growing" if x == 0 else "Divided!" for x in com])
        return cells_futures, cells_state

    def load_current_frame(self, threshold, channels):
        if threshold == -1:
            current_local_data = self.df.loc[(self.df["trench_id"] == self.current_trench)
                                             & (self.df["time_(mins)"] == self.current_frame)].copy()

        else:
            current_local_data = self.df.loc[(self.df["trench_id"] == self.current_trench)
                                             & (self.df["time_(mins)"] == self.current_frame)
                                             & (self.df["centroid-0"] < threshold)].copy()
        # normalise? features do not have the same units - maybe consider only using geometrical features
        # columns = [col for col in self.properties if col not in ["trench_id", "time_(mins)", "label"]]
        # for c in columns:
        #    current_local_data[c] = MinMaxScaler().fit_transform(np.array(current_local_data[c]).reshape(-1, 1))
        cells_list = [Cell(row, channels) for index, row in current_local_data.iterrows()]
        self.current_number_of_cells = len(cells_list)
        return cells_list

    def load_next_frame(self, threshold, channels):
        next_local_data = self.df.loc[(self.df["trench_id"] == self.current_trench)
                                      & (self.df["time_(mins)"] == self.next_frame)
                                      & (self.df["centroid-0"] < threshold)].copy()
        cells_list = [Cell(row, channels) for index, row in next_local_data.iterrows()]
        for i in range(len(cells_list)):
            cells_list[i].set_coordinates(0)
        self.buffer = cells_list
        return cells_list

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

    def calc_score(self, predicted_future, predicted_state, true_coord, max_score, matched_scenario, shift=0, mode="SeqMatch"):
        for i in range(len(predicted_future)):
            # print("the simulated scenario: {}".format(predicted_state[i]))
            pr = predicted_future[i][0]
            # print("with probability: {}".format(pr))
            points = []
            cells_arrangement = []
            for cell in predicted_future[i][1]:
                for c in cell.coord:
                    cells_arrangement.append(cell)
                    points.append([c[0], c[1] - predicted_future[i][1][shift].coord[0][1]])
            points = np.array(points)
            # print(points)
            distance, idx = nearest_neighbour(points, true_coord, mode=mode)
            label_track = []
            for j in idx:
                if j is not None:
                    label_track.append(cells_arrangement[j].label)
                else:
                    label_track.append(None)
            # score = pr / np.sum(distance)
            # for stronger influence of distance
            score = pr / ((np.sum(distance)) ** 2)
            if score > max_score:
                self.next_track = label_track
                max_score = score
                matched_scenario = predicted_state[i]
            # print("score: {}".format(score))
            # print("score_futures result: {}".format(label_track))
            # print(distance)
            # print("\n")
        return max_score, matched_scenario

    def score_futures(self, predicted_future, predicted_state, true_future, mode="SeqMatch", show_details=False):
        # current_points = normalize(current_points, axis=0)
        # current_points[:, 4] = current_points[:, 4] * 20     # add weighting to the centroid_y
        # current_points[:, 0] = current_points[:, 0] * 5      # add weighting to the area
        matched_scenario = []
        true_coord = []
        mcell_current = predicted_future[0][1][0].coord
        for cell in true_future:
            true_coord.append([cell.coord[0][0], cell.coord[0][1] - true_future[0].coord[0][1]])
        true_coord = np.array(true_coord)
        # print(true_coord)
        max_score = 0
        max_score, matched_scenario = self.calc_score(predicted_future, predicted_state, true_coord, max_score,
                                                      matched_scenario, shift=0, mode=mode)
        if abs(true_future[0].coord[0][1] - mcell_current[0][1]) >= \
                ((mcell_current[0][0] + true_future[0].coord[0][0]) * 0.375):
            # the mother cell could have lysed since the first cell shifts more than 3/4 of its length between frames
            print("mother cell lyses or possibly a huge shift in all cells at t = {}min".format(self.current_frame))
            self.calc_score(predicted_future, predicted_state, true_coord,
                            max_score, matched_scenario, shift=1, mode=mode)
        if show_details:
            print("RESULT:")
            print(max_score)
            print(matched_scenario)
            print("\n")

    def lysis_cells(self):
        idx_list = range(self.current_number_of_cells)
        self.current_lysis = [i + 1 for i in idx_list if i + 1 not in self.next_track]

    def track_trench(self, trench, threshold=-1, max_dpf=1, mode="SeqMatch", p_sp=0, special_reporter=None, show_details=False):
        """
        Track cells in specified trench, results are stored in a pandas DataFrame, with a colume that contains
        the labels of the parent cell from previous frame.
        @param trench: trench_id
        @param threshold: the limit of the centroid y axis, this is to limit the number of cells to look at in each trench
        @param max_dpf: is the maximum division per frame to simulate,
        1 or 2 should be enough but in principle this value can go up to the total number of cells below the threshold,
        i.e., all cells divide. If it goes over the total number of cells it will instead use the total number.
        @param mode:  is to select the method used to search the cells' matching future,
        options are simple nearest neighbour 'KDTree'
        or sequence matching 'SeqMatch' (exclusively one-to-one matching, suggested)
        @param p_sp:
        @param special_reporter:
        @param show_details:
        """
        if trench in self.trenches:
            self.current_trench = trench
            frames = self.df.loc[self.df["trench_id"] == self.current_trench, "time_(mins)"].copy()
            frames = sorted(list(set(frames)))
            if threshold == -1:
                threshold = self.max_y
            data_buffer = {
                "track": [],
                "lysis": [],
                "label": [],
                "lysis_frame": [],
                "track_frame": []
            }
            for i in tqdm(range(len(frames) - 1)):
                self.current_frame = frames[i]
                self.next_frame = frames[i + 1]
                self.dt = self.next_frame - self.current_frame
                if special_reporter is None:
                    # Special reporter is to give an ability to track cells using a specified reporter,
                    # e.g., infected by phage
                    # Use the original channels here since no special reporter
                    current_cells = self.load_current_frame(threshold, self.channels)
                    if max_dpf > self.current_number_of_cells:
                        cells_furtures, cells_states = self.cells_simulator(current_cells, self.current_number_of_cells, p_sp=p_sp)
                    else:
                        cells_furtures, cells_states = self.cells_simulator(current_cells, max_dpf, p_sp=p_sp)
                    # this is a list of tuples (probability, cells) and cells is a list of object Cell
                    next_cells = self.load_next_frame(threshold, self.channels)
                    # points = np.array([cell.set_coordinates(0) for cell in next_cells])

                    if show_details:
                        print("looking at cells: ")
                        for x in range(len(current_cells)):
                            print(current_cells[x])

                    self.score_futures(cells_furtures, cells_states, next_cells, mode=mode)
                    # index pointers to the current frame cells from next frame
                    self.lysis_cells()

                    data_buffer["track"].extend(self.next_track)
                    data_buffer["lysis"].extend(self.current_lysis)
                    data_buffer["label"].extend([cell.label for cell in next_cells])
                    data_buffer["lysis_frame"].extend([self.current_frame] * len(self.current_lysis))
                    data_buffer["track_frame"].extend([self.next_frame] * len(self.next_track))

                else:
                    if special_reporter in self.channels:
                        """
                        take the reporter channel out of the channel list, pass to the cell object, 
                        so its data will not be included into the coordinates
                        """
                    else:
                        raise Exception("the specified channel has no data found")
                ### for testing ###
                #if i == 0:
                #    break
            trench_track = [self.current_trench] * len(data_buffer["track_frame"])
            self.track_df = pd.DataFrame(data={
                "trench_id": trench_track,
                "time_(mins)": data_buffer["track_frame"],
                "label": data_buffer["label"],
                "parent_label": data_buffer["track"]
            })
            trench_lyse = [self.current_trench] * len(data_buffer["lysis_frame"])
            self.lysis_df = pd.DataFrame(data={
                "trench_id": trench_lyse,
                "time_(mins)": data_buffer["lysis_frame"],
                "label": data_buffer["lysis"]
            })
