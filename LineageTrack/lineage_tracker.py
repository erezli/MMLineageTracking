import os, re, copy, itertools, pickle, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
# from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.spatial import KDTree, distance_matrix
from scipy.special import owens_t
from scipy.signal import find_peaks
from scipy.stats import linregress, poisson
from scipy.stats import norm, skewnorm, multivariate_normal
from math import comb, isnan, cos
from tqdm import tqdm
from joblib import Parallel, delayed

from LineageTrack.cells import Cell
from LineageTrack.lineages import Lineage


class LineageTrack:
    def __init__(self, df_list, files=None):
        self.files = files
        # Todo: use Zarr array to reduce memory usage
        # Todo: add a trench object?
        self.channels = []
        # list_of_properties = ["trench_id", "time_(mins)", "label", "area", "major_axis_length",
        #                      "minor_axis_length", "centroid-0", "centroid-1",
        #                      "centroid_local-0", "centroid_local-1", "orientation"]
        list_of_properties = ["trench_id", "time_(mins)", "label", "area", "major_axis_length",
                              "minor_axis_length", "centroid-0", "centroid-1", "orientation"]
        if not isinstance(df_list, list):
            df_list = [df_list]
        self.df = df_list[0][list_of_properties].copy()
        for d in df_list:
            channel = d.loc[1, "channel"]
            self.channels.append(channel)
            # self.df.loc[:, "{}_intensity_mean".format(channel)] = d.loc[:, "intensity_mean"]
            self.df.insert(self.df.shape[1], "{}_intensity_mean".format(channel), d.loc[:, "intensity_mean"])
            self.df.insert(self.df.shape[1], "{}_intensity_max".format(channel), d.loc[:, "intensity_max"])
            self.df.insert(self.df.shape[1], "{}_intensity_min".format(channel), d.loc[:, "intensity_min"])
            self.df.insert(self.df.shape[1], "{}_intensity_total".format(channel), d.loc[:, "intensity_total"])
            # might include later but takes a lot of memory
            # self.df.insert(len(self.df.columns),"{}_image_intensity".format(channel), d.loc[:, "image_intensity"])
            # self.df.insert(len(self.df.columns), "{}_identity".format(channel), d.loc[:, "identity"])
            if channel == 'mVenus': # change
                self.df.insert(self.df.shape[1], "{}_intensity_total".format(channel), d.loc[:, "intensity_total"])
            # if channel == 'YFP': # change
            #     self.df.insert(self.df.shape[1], "{}_intensity_total".format(channel), d.loc[:, "intensity_total"])
            if channel == 'PC':
                self.df.insert(self.df.shape[1], "zernike", d.loc[:, "zernike"])
                self.df.insert(self.df.shape[1], "zernike_half", d.loc[:, "zernike_half"])
        self.channels = sorted(list(set(self.channels)))
        self.properties = sorted(list(set(self.df.columns)))
        self.df.sort_values(["trench_id", "time_(mins)", "label"], inplace=True)
        self.trenches = self.df.loc[:, "trench_id"]
        self.trenches = sorted(list(set(self.trenches)))
        self.max_y = self.df.loc[:, "centroid-0"].max()
        self.reporter = None

        # for tracking #
        self.current_trench = 0
        self.current_frame = 0
        self.next_frame = 0
        self.dt = 0
        self.current_number_of_cells = 0
        self.current_cells = None
        self.buffer_next = None
        self.show_details = False
        self.search_mode = "SeqMatch"
        self.probability_mode = "sizer-adder"
        self.dpf = 2
        self.fill_gap = False
        self.drifting = False
        self.skew_model = True
        self.sum_score = None
        self.max_score = None
        self.sec_score = None
        self.sum_prior = None
        self.avg_distance = None
        self.threshold = None
        self.tracked = None

        # model parameters #
        self.div_intervals = []
        self.div_interval = 24  # theoretical time for division
        self.growth_taus = []
        self.growth_tau = (24, 0)
        self.length_at_div = []
        self.d_length = []
        self.sizer_length_paras = (None, None)
        self.sizer_skew = 0
        self.adder_length_paras = (None, None)
        self.adder_skew = 0
        self.mother_cell_collected = []
        self.erode_factor = 1.1  # for fill gaps mode in offset calculation

        # results #
        self.current_track = []
        self.next_track = []
        self.next_track_2 = []
        self.tracked_labels = []
        self.current_lysis = []
        self.current_lysis_2 = None
        self.all_cells = dict()

        print("Finished loading the data")
        # print(f"Head: {self.df.head(1)}")
        print(self.df.shape)

    @classmethod
    def from_path(cls, filepath, *args):
        if os.path.isdir(filepath):
            directory = filepath
            files = glob(directory + "{}*".format(os.path.sep))
        elif os.path.exists(filepath):
            files = [filepath]
        else:
            raise Exception("specified file/directory not found")
        for path in args:
            if not os.path.exists(path):
                print("invalid argument(s) - should be paths to the files or a single directory")
            else:
                files.append(path)
        print("Looking for data at these locations:")
        for f in files:
            print(f)
        cols_list = [list(pd.read_csv(f, nrows=1)) for f in files]
        columns_to_skip = list(["image_intensity", "centroid_local-0", "centroid_local-1"])
        df_list = [pd.read_csv(f, usecols=[i for i in cols if i not in columns_to_skip],
                               dtype={"major_axis_length": np.float32, "minor_axis_length": np.float32,
                                      "centroid-0": np.float32, "centroid-1": np.float32,
                                      "orientation": np.float32, "intensity_mean": np.float32})
                   # converters={"image_intensity": reconstruct_array_from_str})
                   for f, cols in zip(files, cols_list)]
        return cls(df_list=df_list, files=files)

    def __str__(self):
        if self.files:
            return f"""
                Read {len(self.files)} files
                Channels: {self.channels}
                Properties for each cell: {self.properties}
            """
        else:
            return f"""
                Load from Pandas DataFrames
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
            if plot:
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

    def find_division(self, trench, threshold=1, distance=3, height=70, prominence=20, plot=True):
        """
        find and label the division by searching for peaks in the length-time plot, may need user to slice out the bad
        results
        @param trench: the trench id
        @param threshold: for the scipy package find_peaks - the vertical distance to its neighboring samples.
        @param distance: for the scipy package find_peaks - required minimal horizontal distance (>= 1)
        in samples between neighbouring peaks.
        @param plot: plot the peaks and the mother cells growth if set to True
        @return: first entry is a tuple of the trench id and the length-time array of the mother cell, the second entry
        is a 1D array for index of the peaks
        """
        mcell = self.get_mother_cell_growth(trench)
        idx_p = find_peaks(mcell[:, 1], threshold=threshold, distance=distance, height=height, prominence=prominence)
        peaks = [mcell[p, :] for p in idx_p[0]]
        peaks = np.array(peaks)
        if plot:
            plt.figure(figsize=(10, 10))
            plt.plot(mcell[:, 0], mcell[:, 1])
            plt.plot(peaks[:, 0], peaks[:, 1], "o")
            plt.title(f"trench_id: {trench}, the major axis length of the mother cell")
            plt.xlabel("time (min)")
            plt.ylabel("length (px)")
            plt.show()
        mother_cell_info = (trench, mcell)
        return mother_cell_info, idx_p[0]

    def collect_model_para(self, mother_cell, peak_idx, plot=False):
        """
        estimate parameters for the model of cells growth and division in exponential phase
        user defines the period during which cells grow exponentially by examining the data
        *THIS SHOULD ONLY RUN ONCE FOR EACH MOTHER CELL IN EVERY TRENCH*
        @param mother_cell: first entry is the trench id, second entry is the length and time of the mother cell
        @type mother_cell: tuple
        @param peak_idx: sliced index 1D array for exponential phase
        @param plot: whether to plot the model or not
        This collects division time interval, time constant - tau, for the exponential growth model,
        and the max length of cells
        """
        if mother_cell[0] in self.mother_cell_collected:
            return f"""this mother cell at trench {mother_cell[0]} has already been collected"""
        else:
            self.mother_cell_collected.append(mother_cell[0])
            e_peaks = [mother_cell[1][p, :] for p in peak_idx]
            e_peaks = np.array(e_peaks)

            growth_taus = []

            division_intervals = [e_peaks[i + 1][0] - e_peaks[i][0] for i in range(len(e_peaks) - 1)]
            division_times = list(e_peaks[:, 0])
            max_length = list(e_peaks[:, 1])
            d_length = [mother_cell[1][peak_idx[i + 1], 1] - mother_cell[1][peak_idx[i] + 1, 1]
                        for i in range(len(peak_idx) - 1)]

            for i in range(len(peak_idx) - 1):
                growth = mother_cell[1][peak_idx[i] + 1:peak_idx[i + 1] + 1]
                slope, inter, r, p, se = linregress(growth[:, 0] - growth[0, 0], np.log2(growth[:, 1]))
                if plot:
                    plt.plot(growth[:, 0] - growth[0, 0], np.log2(growth[:, 1]))
                    x = np.linspace(0, 24, 100)
                    plt.plot(x, slope * x + inter)
                    print(f"the slope is estimated to be {slope}")
                    print(f"the intercept is estimated to be {inter}")
                    plt.show()
                if not isnan(slope):
                    growth_taus.append(1 / slope)

            self.div_intervals += division_intervals
            self.growth_taus += growth_taus
            self.length_at_div.append([division_times, max_length])
            self.d_length += d_length

    def update_model_para(self, model="unif", Bessel=True):
        # Todo: for cells growing heterogeneously or the growth distribution changes over time - collect model
        #  parameter across mother cells
        if model == "unif":
            self.div_interval = np.mean(self.div_intervals)
            max_lengths = []
            for i in self.length_at_div:
                max_lengths.extend(i[1])
            if Bessel:
                self.growth_tau = (np.mean(self.growth_taus),
                                   np.var(self.growth_taus) * len(self.growth_taus) / (len(self.growth_taus) - 1))
                self.sizer_length_paras = (np.mean(max_lengths),
                                           np.var(max_lengths) * len(max_lengths) / (len(max_lengths) - 1))
                self.sizer_skew = 3 * (np.mean(max_lengths) - np.median(max_lengths)) / np.sqrt(
                    np.var(max_lengths) * len(max_lengths) / (len(max_lengths) - 1))
                self.adder_length_paras = (np.mean(self.d_length),
                                           np.var(self.d_length) * len(self.d_length) / (len(self.d_length) - 1))
                self.adder_skew = 3 * (np.mean(self.d_length) - np.median(self.d_length)) / np.sqrt(
                    np.var(self.d_length) * len(self.d_length) / (len(self.d_length) - 1))
            else:
                self.growth_tau = (np.mean(self.growth_taus), np.var(self.growth_taus))
                self.sizer_length_paras = (np.mean(max_lengths), np.var(max_lengths))
                self.sizer_skew = 3 * (np.mean(max_lengths) - np.median(max_lengths)) / np.sqrt(np.var(max_lengths))
                self.adder_length_paras = (np.mean(self.d_length), np.var(self.d_length))
                self.adder_skew = 3 * (np.mean(self.d_length) - np.median(self.d_length)) / np.sqrt(
                    np.var(self.d_length))
            print(f"""
                    The average time interval for division is {self.div_interval}
                    The time constant for exponential growth is {self.growth_tau}
                    The average division length is {self.sizer_length_paras[0]} 
                    with variance {self.sizer_length_paras[1]} and skewness {self.sizer_skew}
                    The length for adder model is {self.adder_length_paras[0]} 
                    with variance {self.adder_length_paras[1]} and skewness {self.adder_skew}
                    """)
        elif model == "lineage-trench":
            max_lengths = []
            adder_lengths = []
            timer_periods = []
            growth_time_consts = []

            lineages = self.generate_lineage(self.current_trench)
            max_lengths += [line.lengths[-1] for line in lineages if line.daughters[0] is not None]
            adder_lengths += [line.get_adder_dl() for line in lineages if line.get_adder_dl() is not None]
            timer_periods += [line.get_timer_dt() for line in lineages if line.get_timer_dt() is not None]
            q75, q25 = np.percentile(self.growth_taus, [75, 25])
            intr_qr = q75 - q25
            max_thr = q75 + (1.5 * intr_qr)
            min_thr = q25 - (1.5 * intr_qr)
            growth_time_consts += [line.get_growth_time_constant() for line in lineages
                                   if line.get_growth_time_constant() is not None and
                                   max_thr > line.get_growth_time_constant() > min_thr]
            self.div_interval = np.mean(timer_periods)
            if len(growth_time_consts) > 2:
                if Bessel:
                    self.growth_tau = (np.mean(growth_time_consts),
                                       np.var(growth_time_consts) *
                                       len(growth_time_consts) / (len(growth_time_consts) - 1))
                else:
                    self.growth_tau = (np.mean(growth_time_consts), np.var(growth_time_consts))
            if len(max_lengths) <= 2:
                max_lengths = []
                for i in self.length_at_div:
                    max_lengths.extend(i[1])
            if Bessel:
                self.sizer_length_paras = (np.mean(max_lengths),
                                           np.var(max_lengths) * len(max_lengths) / (len(max_lengths) - 1))
                self.sizer_skew = 3 * (np.mean(max_lengths) - np.median(max_lengths)) / np.sqrt(
                    np.var(max_lengths) * len(max_lengths) / (len(max_lengths) - 1))
            else:
                self.sizer_length_paras = (np.mean(max_lengths), np.var(max_lengths))
                self.sizer_skew = 3 * (np.mean(max_lengths) - np.median(max_lengths)) / np.sqrt(np.var(max_lengths))
            if len(adder_lengths) > 2:
                if Bessel:
                    self.adder_length_paras = (np.mean(adder_lengths),
                                               np.var(adder_lengths) * len(adder_lengths) / (len(adder_lengths) - 1))
                    self.adder_skew = 3 * (np.mean(adder_lengths) - np.median(adder_lengths)) / np.sqrt(
                        np.var(adder_lengths) * len(adder_lengths) / (len(adder_lengths) - 1))
                else:
                    self.adder_length_paras = (np.mean(adder_lengths), np.var(adder_lengths))
                    self.adder_skew = 3 * (np.mean(adder_lengths) - np.median(adder_lengths)) / np.sqrt(
                        np.var(adder_lengths))

            print(f"""
                    The average time interval for division is {self.div_interval}
                    The time constant for exponential growth is {self.growth_tau}
                    The average division length is {self.sizer_length_paras[0]} 
                    with variance {self.sizer_length_paras[1]} and skewness {self.sizer_skew}
                    The length for adder model is {self.adder_length_paras[0]} 
                    with variance {self.adder_length_paras[1]} and skewness {self.adder_skew}
                    """)
        elif model == "lineage-all":
            max_lengths = []
            adder_lengths = []
            timer_periods = []
            growth_time_consts = []
            for key in self.all_cells:
                lineages = self.generate_lineage(key)
                max_lengths += [line.lengths[-1] for line in lineages if line.daughters[0] is not None]
                adder_lengths += [line.get_adder_dl() for line in lineages if line.get_adder_dl() is not None]
                timer_periods += [line.get_timer_dt() for line in lineages if line.get_timer_dt() is not None]
                growth_time_consts += [line.get_growth_time_constant() for line in lineages
                                       if line.get_growth_time_constant() is not None]
            self.div_interval = np.mean(timer_periods)
            if Bessel:
                self.growth_tau = (np.mean(growth_time_consts),
                                   np.var(growth_time_consts) * len(growth_time_consts) / (len(growth_time_consts) - 1))
                self.sizer_length_paras = (np.mean(max_lengths),
                                           np.var(max_lengths) * len(max_lengths) / (len(max_lengths) - 1))
                self.sizer_skew = 3 * (np.mean(max_lengths) - np.median(max_lengths)) / np.sqrt(
                    np.var(max_lengths) * len(max_lengths) / (len(max_lengths) - 1))
                self.adder_length_paras = (np.mean(adder_lengths),
                                           np.var(adder_lengths) * len(adder_lengths) / (len(adder_lengths) - 1))
                self.adder_skew = 3 * (np.mean(adder_lengths) - np.median(adder_lengths)) / np.sqrt(
                    np.var(adder_lengths) * len(adder_lengths) / (len(adder_lengths) - 1))
            else:
                self.growth_tau = (np.mean(growth_time_consts), np.var(growth_time_consts))
                self.sizer_length_paras = (np.mean(max_lengths), np.var(max_lengths))
                self.sizer_skew = 3 * (np.mean(max_lengths) - np.median(max_lengths)) / np.sqrt(np.var(max_lengths))
                self.adder_length_paras = (np.mean(adder_lengths), np.var(adder_lengths))
                self.adder_skew = 3 * (np.mean(adder_lengths) - np.median(adder_lengths)) / np.sqrt(
                    np.var(adder_lengths))
            print(f"""
                    The average time interval for division is {self.div_interval}
                    The time constant for exponential growth is {self.growth_tau}
                    The average division length is {self.sizer_length_paras[0]} 
                    with variance {self.sizer_length_paras[1]} and skewness {self.sizer_skew}
                    The length for adder model is {self.adder_length_paras[0]} 
                    with variance {self.adder_length_paras[1]} and skewness {self.adder_skew}
                    """)

    def calculate_growth(self):
        return 2 ** (self.dt / self.growth_tau[0])

    def pr_div_lambda(self, n):
        # deprecated
        # n = 0: no division
        # n = 1: division
        return poisson.pmf(n, self.dt / self.growth_tau[0])

    def pr_div_sizer(self, n, length):
        if self.skew_model:
            # cdf = skewnorm.cdf(length, np.array([self.sizer_skew] * len(n)), loc=self.sizer_length_paras[0],
            #                    scale=np.sqrt(self.sizer_length_paras[1]))
            # scipy skewnorm.cdf is extremely slow because they used integrate, this is fixed in 1.10.0 milestone
            # however the 1.10.0 version is not released yet
            # refer to Amsler, C., Papadopoulos, A. & Schmidt, P. Evaluating the cdf of the Skew Normal distribution.
            cdf = sn_cdf(length, self.sizer_skew, self.sizer_length_paras[0], self.sizer_length_paras[1])
        else:
            cdf = norm.cdf(length, loc=self.sizer_length_paras[0], scale=np.sqrt(self.sizer_length_paras[1]))
        return np.prod(cdf * n + (1 - cdf) * (1 - n))

    def pr_div_adder(self, n, dl):
        if self.adder_length_paras[0] and self.adder_length_paras[1] is not None:
            if self.skew_model:
                # cdf = skewnorm.cdf(dl, np.array([self.adder_skew] * len(n)), loc=self.adder_length_paras[0],
                #                    scale=np.sqrt(self.adder_length_paras[1]))
                cdf = sn_cdf(dl, self.adder_skew, self.adder_length_paras[0], self.adder_length_paras[1])
            else:
                cdf = norm.cdf(dl, loc=self.adder_length_paras[0], scale=np.sqrt(self.adder_length_paras[1]))
            return np.prod(cdf * n + (1 - cdf) * (1 - n))
        else:
            return 1

    def pr_div_sizer_adder(self, n, dl, length):
        if self.adder_length_paras[0] and self.adder_length_paras[1] is not None:
            if self.skew_model:
                # cdf = skewnorm.cdf(dl, np.array([self.adder_skew] * len(n)), loc=self.adder_length_paras[0],
                #                    scale=np.sqrt(self.adder_length_paras[1]))
                cdf1 = sn_cdf(dl, self.adder_skew, self.adder_length_paras[0], self.adder_length_paras[1])
                cdf2 = sn_cdf(length, self.sizer_skew, self.sizer_length_paras[0], self.sizer_length_paras[1])
            else:
                cdf1 = norm.cdf(dl, loc=self.adder_length_paras[0], scale=np.sqrt(self.adder_length_paras[1]))
                cdf2 = norm.cdf(length, loc=self.sizer_length_paras[0], scale=np.sqrt(self.sizer_length_paras[1]))
            return np.prod(cdf1 * cdf2 * n + (1 - cdf1 * cdf2) * (1 - n))
        else:
            return self.pr_div_sizer(n, length)

    def cells_simulator(self, cells_list, p_sp, radius=0):
        growth = self.calculate_growth()

        # if self.probability_mode == "sizer-adder" and self.current_frame > self.div_interval / 2:
        #     dl = []
        #     for i in range(self.current_number_of_cells):
        #         lineage_parent = cells_list[i]
        #         while lineage_parent.parent is not None and lineage_parent.parent.divide is False:
        #             lineage_parent = lineage_parent.parent
        #         dl.append(cells_list[i].major - lineage_parent.major)
        #     prob = self.pr_div_sizer_adder(np.array([0] * len(cells_list)), dl, [c.major for c in cells_list])
        # else:
        #     prob = self.pr_div_sizer(np.array([0] * len(cells_list)), [c.major for c in cells_list])

        cells_future = [p_sp, cells_list]
        if self.tracked:
            additional = self.tracked[3][0]
        else:
            additional = 0
        cells_state = ["SP"] * (len(cells_list) + additional)
        yield cells_future, cells_state
        for d in range(self.dpf + 1):
            d_list = []
            d_list.extend([1] * d)
            if self.tracked:
                d_list.extend([0] * (self.current_number_of_cells - len(self.tracked[0]) - d))
            else:
                d_list.extend([0] * (self.current_number_of_cells - d))
            # combinations = set(list(itertools.permutations(d_list))) # very wasteful
            for com in binary_permutations(d_list):
                lengths = []
                dl = []
                offset = 0
                if self.tracked:
                    com_whole = self.tracked[0] + com
                    len_tracked = len(self.tracked[0])
                else:
                    com_whole = com
                    len_tracked = 0
                for i in range(self.current_number_of_cells - len_tracked):
                    cells_list[i].set_coordinates(division=com[i], growth=growth, offset=offset, radius=radius)
                    if cells_list[i].coord[-1][1] < self.threshold * 0.9:
                        cells_list[i].within_safe_zone = True
                    if self.probability_mode == "sizer-adder" and self.current_frame > self.div_interval / 2:
                        lineage_parent = cells_list[i]
                        while lineage_parent.parent is not None and lineage_parent.parent.divide is False:
                            lineage_parent = lineage_parent.parent
                        dl.append(cells_list[i].major - lineage_parent.major)
                    offset += cells_list[i].major * (growth - 1) * cos(
                        cells_list[i].orientation)  # / self.erode_factor
                    # prob *= prob_0 * self.pr_div_sizer(0, cells_list[i].coord[0][0])
                    # prob *= prob_1 * self.pr_div_sizer(1, cells_list[i].coord[0][0] * 2)
                    lengths.append(cells_list[i].major * growth)
                    if i != self.current_number_of_cells - len_tracked - 1 and self.fill_gap is True:
                        if (cells_list[i].major + cells_list[i + 1].major) * growth / 2 * 1.1 < \
                                (cells_list[i + 1].centroid_y - cells_list[i].centroid_y):
                            offset = max(0, offset - ((cells_list[i + 1].centroid_y - cells_list[i].centroid_y) -
                                                      (cells_list[i].major + cells_list[i + 1].major) * growth / 2)
                                         * self.erode_factor)  # Todo: factor for segmentation erosion
                prob = self.pr_div_sizer(np.array(com), lengths)
                if self.probability_mode == "sizer-adder" and self.current_frame > self.div_interval / 2:
                    prob = self.pr_div_sizer_adder(np.array(com), dl, lengths)
                # Todo: probability of division - dependency between cells in a trench
                cells_future = [prob, cells_list]
                cells_state = ["Growing" if x == 0 else "Divided!" for x in com_whole]
                yield cells_future, cells_state
        # these cells_futures is a list of tuples (probability, cells) and cells is a list of object Cell
        # return cells_futures, cells_states

    def load_current_frame(self, radius=0):
        if self.buffer_next:
            if self.tracked:
                cells_to_add = self.tracked[5]
            else:
                cells_to_add = []
            trench_cells = cells_to_add + self.buffer_next
            cells_list = trench_cells
        else:
            current_local_data = self.df.loc[(self.df["trench_id"] == self.current_trench)
                                             & (self.df["time_(mins)"] == self.current_frame)
                                             & (self.df["centroid-0"] < self.threshold)].copy()
            # normalise? features do not have the same units - maybe consider only using geometrical features
            # columns = [col for col in self.properties if col not in ["trench_id", "time_(mins)", "label"]]
            # for c in columns:
            #    current_local_data[c] = MinMaxScaler().fit_transform(np.array(current_local_data[c]).reshape(-1, 1))
            if self.reporter:
                cells_list = [Cell(row, self.channels, reporter=self.reporter)
                              for index, row in current_local_data.iterrows()]
            else:
                cells_list = [Cell(row, self.channels) for index, row in current_local_data.iterrows()]
            for i in range(len(cells_list)):
                cells_list[i].set_coordinates(radius=radius)
        self.current_number_of_cells = len(cells_list)
        self.current_cells = cells_list

    def load_next_frame(self, radius=0):
        next_local_data = self.df.loc[(self.df["trench_id"] == self.current_trench)
                                      & (self.df["time_(mins)"] == self.next_frame)
                                      & (self.df["centroid-0"] < self.threshold)].copy()
        if self.reporter:
            cells_list = [Cell(row, self.channels, reporter=self.reporter) for index, row in next_local_data.iterrows()]
        else:
            cells_list = [Cell(row, self.channels) for index, row in next_local_data.iterrows()]
        for i in range(len(cells_list)):
            cells_list[i].set_coordinates(radius=radius)
        self.buffer_next = cells_list
        # return cells_list

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

    def nearest_neighbour(self, points, true_coord):
        if self.search_mode == "KDTree":
            grid = KDTree(points)
            # idx points to the index of points constructing the KDTree
            distance, idx = grid.query(true_coord, workers=-1)
            return distance, idx
        if self.search_mode == "SeqMatch":
            # match exclusively
            d_mtx = distance_matrix(true_coord, points)
            distance = []
            idx = []
            index_mtx = np.argsort(d_mtx, axis=1)
            for row in range(index_mtx.shape[0]):
                for col in range(index_mtx.shape[1]):
                    if len(idx) == 0:
                        idx.append(index_mtx[row][col])
                        distance.append(d_mtx[row][index_mtx[row][col]])
                        break
                    elif index_mtx[row][col] > np.nanmax(np.array(idx, dtype=np.float64)):
                        idx.append(index_mtx[row][col])
                        distance.append(d_mtx[row][index_mtx[row][col]])
                        break
                    elif col == index_mtx.shape[1] - 1:
                        # for the case next frame has more cells than the simulated scenario
                        # if self.current_frame == 84:
                        #     print(row)
                        #     print(col)
                        #     print(index_mtx)
                        #     print(idx)
                        if len(idx) - sum(x is None for x in idx) == index_mtx.shape[1]:
                            distance.append(d_mtx[row][col])
                        else:
                            miss = [index for index in range(index_mtx.shape[1]) if index not in idx]
                            distance.append(d_mtx[row][miss[-1]])
                        idx.append(None)
            if idx != [x for x in range(len(idx))]:
                distance2 = np.diagonal(d_mtx)
                # print("for fake lysis: {}".format(np.mean(distance2)))
                if np.mean(distance2) < np.mean(distance):
                    distance = distance2
                    # idx = [x if idx[x] is not None else None for x in range(len(idx))]
                    idx = [x if x < index_mtx.shape[1] else None for x in range(len(idx))]
                    # print("fake lysis")
            return distance, idx

    def store_cells_info(self, cells):
        # Todo: only update the untracked and truncated cells
        if self.tracked:
            cells_to_add = self.tracked[4]
        else:
            cells_to_add = []
        trench_cells = cells_to_add + cells

        for i in range(len(trench_cells)):
            trench_cells[i].set_coordinates(reset_original=True)

        if self.i != 0:  # not the first frame
            # update previous frame cells from the previous tracking results
            for cell in self.all_cells[self.current_trench][self.i - 1]:
                cell.daughters = None
            for cell, parent_label in zip(trench_cells, self.current_track):
                if parent_label is not None:
                    cell.parent_label = parent_label
                    cell.set_parent(self.all_cells[self.current_trench][self.i - 1][int(parent_label - 1)])
                    self.all_cells[self.current_trench][self.i - 1][int(parent_label - 1)].set_daughters(cell)
            # assign barcode and poles to some cells
            for cell in self.all_cells[self.current_trench][self.i - 1]:
                try:
                    cell.assign_barcode(to_whom="daughter", max_bit=8)
                except TypeError:
                    print(f"cell {cell.label} at time {cell.time} failed assigning barcode (could have been converted already)")
                cell.barcode_to_binary(max_bit=8)
                cell.set_generation()
        else:
            # initialise the barcode for the very first two cells in the first frame
            trench_cells[0].barcode = list([0])
            if len(trench_cells) > 1:
                trench_cells[1].barcode = list([1])
            trench_cells[0].poles = (0, 0)
        if self.probability_mode == "sizer-adder":
            for cell_next, parent_label in zip(self.buffer_next, self.next_track):
                if parent_label is not None:
                    cell_next.set_parent(trench_cells[int(parent_label) - 1])
        # if self.tracked:
        if len(self.all_cells[self.current_trench]) > self.i: # a bit hacky, ideally want to check if this frame is tracked. Be careful when there is only one cell
            self.all_cells[self.current_trench][self.i] = trench_cells
        else:
            self.all_cells[self.current_trench].append(trench_cells)

    def calc_score(self, cells_list, p_sp, true_coord, matched_scenario, matched_scenario_2, cells, cells_2, shift=0, radius=0):
        # Todo: using index will be wrong for cases tracked cells are removed
        # for i in range(len(predicted_future)):
        for predicted_future, predicted_state in self.cells_simulator(cells_list, p_sp=p_sp, radius=radius):
            pr = predicted_future[0]
            points = []
            cells_arrangement = []
            if self.drifting:
                drift = np.mean([cell.centroid_y for cell in predicted_future[1]])
            elif self.tracked:
                drift = self.tracked[1][0]
            elif shift < len(predicted_future[1]):
                drift = predicted_future[1][shift].coord[0][1]
            else:
                drift = predicted_future[1][-1].coord[0][1]
            # Todo: if cumulative here, the drift should the be the bottom of the last tracked cell
            for cell in predicted_future[1]:
                for c in cell.coord:
                    cells_arrangement.append(cell)
                    points.append([c[0], (c[1] - drift), c[2], c[3]])
            points = np.array(points)
            distance, idx = self.nearest_neighbour(points, true_coord)
            # score = pr / np.sum(distance)
            # for stronger influence of distance
            try:
                # score = (0.3 + pr ** 0.5) / ((np.sum(distance)) ** 3)
                score = scoring(pr, distance)
                self.sum_score += score
                if pr != -1:
                    self.sum_prior += pr
                self.avg_distance += np.mean(distance)
            except RuntimeWarning:
                print("RuntimeWarning - consider divide by zero error")
                print((np.mean(distance)))
                score = float("inf")
                label_track = []
                for j in idx:
                    if j is not None:
                        label_track.append(cells_arrangement[j].label)
                    else:
                        label_track.append(None)
                self.next_track = label_track
                self.max_score = score
                matched_scenario = predicted_state
                cells = copy.deepcopy(predicted_future[1])
                break
            if score >= self.max_score:
                self.sec_score = self.max_score
                label_track = []
                for j in idx:
                    if j is not None:
                        label_track.append(cells_arrangement[j].label)
                    else:
                        label_track.append(None)
                self.next_track_2 = self.next_track
                matched_scenario_2 = matched_scenario
                cells_2 = copy.deepcopy(cells)
                self.next_track = label_track
                self.max_score = score
                matched_scenario = predicted_state
                cells = copy.deepcopy(predicted_future[1])
            elif score > self.sec_score:
                self.sec_score = score
                label_track = []
                for j in idx:
                    if j is not None:
                        label_track.append(cells_arrangement[j].label)
                    else:
                        label_track.append(None)
                self.next_track_2 = label_track
                matched_scenario_2 = predicted_state
                cells_2 = copy.deepcopy(predicted_future[1])
            if self.show_details:
                print("the simulated scenario: {}".format(predicted_state))
                print([ll.label for ll in cells_arrangement])
                print(idx)
                print(points)
                print("with probability: {}".format(pr))
                print("score: {}".format(score))
                # print("score_futures result: {}".format(label_track))
                print(np.mean(distance))
                # print("\n")
        return (matched_scenario, matched_scenario_2), (cells, cells_2)

    def score_futures(self, p_sp=0, radius=0):
        # current_points = normalize(current_points, axis=0)
        # current_points[:, 4] = current_points[:, 4] * 20     # add weighting to the centroid_y
        # current_points[:, 0] = current_points[:, 0] * 5      # add weighting to the area
        true_coord = []
        matched_scenario = []
        matched_scenario_2 = []
        cells = None
        cells_2 = None
        mcell_current = self.current_cells[0].coord
        if self.drifting:
            drift = np.mean([cell.centroid_y for cell in self.buffer_next])
        elif self.tracked:
            drift = self.tracked[1][1]
        else:
            drift = self.buffer_next[0].coord[0][1]
        for cell in self.buffer_next:
            true_coord.append([cell.coord[0][0], (cell.coord[0][1] - drift), cell.coord[0][2], cell.coord[0][3]])
        true_coord = np.array(true_coord)
        self.max_score = 0
        self.sec_score = 0
        self.sum_score = 0
        self.sum_prior = 0
        self.avg_distance = 0
        (matched_scenario, matched_scenario_2), (cells, cells_2) = self.calc_score(self.current_cells, p_sp, true_coord,
                                                                                   matched_scenario, matched_scenario_2,
                                                                                   cells, cells_2, shift=0, radius=radius)
        # Todo: Dont do this if eliminated tracked
        if abs(self.buffer_next[0].coord[0][1] - mcell_current[0][1]) >= \
                ((mcell_current[0][0] + self.buffer_next[0].coord[0][0]) * 0.375) and not self.tracked:
            # the mother cell could have lysed since the first cell shifts more than 3/4 of its length between frames
            arr = np.asarray([c[1] for cell in self.current_cells for c in cell.coord])
            index = (np.abs(arr - self.buffer_next[0].coord[0][1])).argmin()
            if self.show_details:
                print("mother cell lyses or possibly a huge shift in all cells at t = {}min".format(self.current_frame))
                print("shift by {} cells".format(index))
            (matched_scenario, matched_scenario_2), (cells, cells_2) = self.calc_score(self.current_cells, p_sp,
                                                                                       true_coord, matched_scenario,
                                                                                       matched_scenario_2, cells,
                                                                                       cells_2, shift=index, radius=radius)
        confidence = self.max_score / self.sum_score
        confidence_2 = self.sec_score / self.sum_score
        if self.show_details:
            print(true_coord)
            print("RESULT:")
            print(self.max_score)
            print("Confidence: {}".format(confidence))
            print(matched_scenario)
            print("\n")
        for i in range(len(cells)):
            cells[i].set_coordinates(reset_original=True)
        return cells, confidence, cells_2, confidence_2

    def lysis_cells(self):
        # Todo: tracked cell from previous iter will not be in the list
        # if self.tracked:
        if len(self.all_cells[self.current_trench]) > self.i: # a bit hacky, ideally want to check if this frame is tracked. Be careful when there is only one cell:
            try:
                offset = self.tracked[3][0]
            except:
                offset = 0
            frame_idx = self.i
        else:
            offset = 0
            frame_idx = -1
        idx_list = range(offset, self.current_number_of_cells)
        self.current_lysis = [i + 1 for i in idx_list if i + 1 not in self.next_track]
        for lysis in self.current_lysis:
            self.all_cells[self.current_trench][frame_idx][lysis - 1].out = True
            if self.all_cells[self.current_trench][frame_idx][lysis - 1].within_safe_zone:
                self.all_cells[self.current_trench][frame_idx][lysis - 1].lyse = True

    def lysis_cells_2(self):
        # Todo: tracked cell from previous iter will not be in the list
        # if self.tracked:
        if len(self.all_cells[self.current_trench]) > self.i: # a bit hacky, ideally want to check if this frame is tracked. Be careful when there is only one cell:
            try:
                offset = self.tracked[3][0]
            except:
                offset = 0
        else:
            offset = 0
        idx_list = range(offset, self.current_number_of_cells)
        self.current_lysis_2 = [i + 1 for i in idx_list if i + 1 not in self.next_track_2]

    def track_trench(self, trench, threshold=-1, max_dpf=2, search_mode="SeqMatch", probability_mode="sizer-adder",
                     p_sp=-1, special_reporter=None, show_details=False, ret_df=True, fill_gap=False, adap_dpf=False,
                     drift=False, skew_model=True, update_para=False, cumulative=True, radius=0):
        """
        Track cells in specified trench, results are stored in a pandas DataFrame, with a colume that contains
        the labels of the parent cell from previous frame.
        
        @param trench: trench_id
        @param threshold: the limit of the centroid y-axis, this is to limit the number of cells to look at in each trench
        @param max_dpf: the maximum division per frame to simulate,
        1 or 2 should be enough but in principle this value can go up to the total number of cells below the threshold,
        i.e., all cells divide. If it goes over the total number of cells it will instead use the total number.
        @param search_mode: to select the method used to search the cells' matching future,
        options are simple nearest neighbour 'KDTree'
        or sequence matching 'SeqMatch' (exclusively one-to-one matching, suggested)
        @param probability_mode: sizer, or a combination of sizer and adder 
        @param p_sp: Probability of all cells entering stationary phase, i.e., stop growing.
        @param special_reporter: e.g. YFP - special channel intensities to keep track
        @param show_details: Display more details about the process
        @param ret_df: If True, output pandas dataframe. If False, output dictionary structure.
        @param fill_gap:
        @param adap_dpf:
        @param drift:
        @param cumulative:
        @param skew_model:
        @param update_para:
        """
        if trench in self.trenches:
            self.current_trench = int(trench)
            self.threshold = threshold
            self.buffer_next = None
            self.show_details = show_details
            self.search_mode = search_mode
            self.probability_mode = probability_mode
            self.fill_gap = fill_gap
            self.drifting = drift
            self.skew_model = skew_model
            self.tracked = None
            frames = self.df.loc[self.df["trench_id"] == self.current_trench, "time_(mins)"].copy()
            frames = sorted(list(set(frames)))
            if threshold == -1:
                self.threshold = self.max_y
            if special_reporter in self.channels:
                self.reporter = special_reporter
                # self.channels.remove(special_reporter)
            elif special_reporter is not None and special_reporter is not self.reporter:
                print(self.channels)
                print(special_reporter)
                raise Exception("the specified channel has no data found")
            data_buffer = {
                "trench": trench,
                "track-1": [],
                "confidence-1": [],
                "track-2": [],
                "confidence-2": [],
                "lysis-1": [],
                "lysis-2": [],
                "label": [],
                "lysis_frame": [],
                "track_frame": [],
                "coord": [],
                "barcode": [],
                "poles": []
            }
            if self.current_trench not in self.all_cells or cumulative == False:
                self.all_cells[self.current_trench] = []
            number_cells = []
            for i in tqdm(range(len(frames) - 1), desc=f"Tracking over frames in trench {trench}: "):
                self.i = i
                self.current_frame = frames[i]
                self.next_frame = frames[i + 1]
                self.dt = self.next_frame - self.current_frame
                self.current_track = self.tracked_labels + self.next_track

                self.load_current_frame()
                if self.show_details:
                    print("cells before cumulative check: ")
                    print(self.current_cells)
                    for x in range(len(self.current_cells)):
                        print(self.current_cells[x])
                self.load_next_frame()
                # points = np.array([cell.set_coordinates(0) for cell in self.buffer_next])
                if adap_dpf:
                    self.dpf = max_dpf + max(0, len(self.buffer_next) - self.current_number_of_cells)
                else:
                    self.dpf = max_dpf
                self.tracked_labels = []
                current_cells_tracked = []
                buffer_next_tracked = []
                if cumulative and i + 1 < len(self.all_cells[self.current_trench]):
                    tracked_cells = []
                    for n in range(len(self.all_cells[self.current_trench][i])):
                        if self.all_cells[self.current_trench][i][-n - 1].daughters is not None:
                            if n == 0:
                                tracked_cells = self.all_cells[self.current_trench][i][:]
                            else:
                                tracked_cells = self.all_cells[self.current_trench][i][:-n]
                            break
                    comb_list = [0 if cell.divide is False else 1 for cell in tracked_cells]
                    if len(tracked_cells) > 1:
                        truncated = False
                        drift_current_frame = None
                        no_current_tracked = 0
                        drift_next_frame = None
                        no_next_tracked = 0
                        for cell in reversed(tracked_cells):
                            if cell.divide and not isinstance(cell.daughters, tuple):
                                truncated = True
                                pass
                            elif cell.daughters is None:
                                pass
                            else:
                                drift_current_frame = cell.centroid_y + cell.major / 2
                                no_current_tracked = int(cell.label)
                                if isinstance(cell.daughters, tuple):
                                    drift_next_frame = cell.daughters[-1].centroid_y + cell.daughters[-1].major / 2
                                    no_next_tracked = int(cell.daughters[-1].label)
                                else:
                                    drift_next_frame = cell.daughters.centroid_y + cell.daughters.major / 2
                                    no_next_tracked = int(cell.daughters.label)
                                self.tracked_labels = [c.parent.label for c in
                                                       self.all_cells[self.current_trench][i + 1][:no_next_tracked]]
                                break
                        # Todo: remove the truncated lineage from the all_cells, remove the fully tracked cells from
                        #  current and next frame
                        self.current_cells = self.current_cells[no_current_tracked:]
                        buffer_next_tracked = self.buffer_next[:no_next_tracked]
                        self.buffer_next = self.buffer_next[no_next_tracked:]

                        self.tracked = (comb_list[:no_current_tracked],
                                        (drift_current_frame, drift_next_frame),
                                        truncated,
                                        (no_current_tracked, no_next_tracked),
                                        self.all_cells[self.current_trench][i][:no_current_tracked],
                                        self.all_cells[self.current_trench][i + 1][:no_next_tracked])
                        # print(self.tracked)
                        if drift_current_frame is None:
                            self.tracked = None
                    else:
                        self.tracked = None
                    # no_untracked = self.current_number_of_cells - max(len(tracked_cells)-1, 0)
                    no_untracked = len(self.current_cells)
                    if self.dpf > no_untracked or max_dpf == -1:
                        self.dpf = no_untracked
                else:
                    if self.dpf > self.current_number_of_cells or max_dpf == -1:
                        self.dpf = self.current_number_of_cells
                if self.show_details:
                    print("looking at cells: ")
                    for x in range(len(self.current_cells)):
                        print(self.current_cells[x])
                if len(self.buffer_next) == 0:
                    print("NO cells to track at {}".format(self.current_frame))
                    self.next_track = []
                    self.next_track_2 = []
                    # if len(self.current_cells) != 0:
                    cells = copy.deepcopy(self.current_cells)
                    self.store_cells_info(cells)
                    confidence = 1
                    confidence_2 = 0
                    self.lysis_cells()
                    self.lysis_cells_2()
                elif len(self.current_cells) == 0:
                    print("NO cells presents at {}".format(self.current_frame))
                    cells = copy.deepcopy(self.current_cells)
                    self.store_cells_info(cells)
                    confidence = 1
                    confidence_2 = 0
                    self.lysis_cells()
                    self.lysis_cells_2()
                else:
                    cells, confidence, cells_2, confidence_2 = self.score_futures(p_sp=p_sp, radius=radius)
                    number_cells.append(self.current_number_of_cells)
                    # print("confidence: {}".format(confidence))
                    # print(self.sum_score)
                    # if confidence < 0.5:
                    #     print("\t second confidence: {}".format(confidence_2))
                    # print(self.sum_prior)
                    # print(self.avg_distance)
                    self.store_cells_info(cells)
                    # index pointers to the current frame cells from next frame
                    self.lysis_cells()
                    self.lysis_cells_2()

                # Todo: Cell simulations missing tracked
                if i == 0:
                    data_buffer["track-1"].append([None] * self.current_number_of_cells)
                    data_buffer["confidence-1"].append(None)
                    data_buffer["track-2"].append([None] * self.current_number_of_cells)
                    data_buffer["confidence-2"].append(None)
                    data_buffer["track_frame"].append(self.current_frame)  # * self.current_number_of_cells)
                    data_buffer["coord"].append([(cell.centroid_x, cell.centroid_y)
                                                 for cell in self.all_cells[self.current_trench][0]])
                else:
                    for cell in self.all_cells[self.current_trench][i - 1]:
                        cell.barcode_to_binary(max_bit=8)
                    data_buffer["barcode"].append([cell.barcode for cell in self.all_cells[self.current_trench][i - 1]])
                    data_buffer["poles"].append([cell.poles for cell in self.all_cells[self.current_trench][i - 1]])

                data_buffer["label"].append([cell.label
                                             for cell in self.all_cells[self.current_trench][i]])
                data_buffer["track-1"].append(self.tracked_labels + self.next_track)
                data_buffer["confidence-1"].append(confidence)
                data_buffer["track-2"].append(self.tracked_labels + self.next_track_2)
                data_buffer["confidence-2"].append(confidence_2)

                data_buffer["track_frame"].append(self.next_frame)  # * len(self.next_track))
                data_buffer["coord"].append([(cell.centroid_x, cell.centroid_y)
                                             for cell in buffer_next_tracked + self.buffer_next
                                             if self.buffer_next is not None])

                data_buffer["lysis-1"].append(self.current_lysis)
                data_buffer["lysis-2"].append(self.current_lysis_2)
                data_buffer["lysis_frame"].append(self.current_frame)  # * len(self.current_lysis))

                # for testing #
                # if i == 27:
                #     self.show_details = True
                # elif i == 29:
                #     self.show_details = False

            barcode_list = []
            poles_list = []
            assert isinstance(self.buffer_next, list)
            if self.tracked:
                cells_to_add = self.tracked[5]
            else:
                cells_to_add = []
            trench_cells = cells_to_add + self.buffer_next
            for cell in self.all_cells[self.current_trench][self.i]:
                cell.daughters = None
            for cell, parent_label in zip(trench_cells, self.tracked_labels + self.next_track):
                if parent_label is not None:
                    cell.parent_label = parent_label
                    cell.set_parent(self.all_cells[self.current_trench][self.i][int(parent_label - 1)])
                    self.all_cells[self.current_trench][self.i][int(parent_label - 1)].set_daughters(cell)
            if len(self.all_cells[self.current_trench][self.i]) != 0:
                for cell in self.all_cells[self.current_trench][self.i]:
                    try:
                        cell.assign_barcode(to_whom="daughter", max_bit=8)
                    except TypeError:
                        print(f"cell {cell.label} at time {cell.time}")
                    cell.barcode_to_binary(max_bit=8)
                    cell.set_generation()
            data_buffer["barcode"].append([cell.barcode for cell in self.all_cells[self.current_trench][self.i]])
            data_buffer["poles"].append([cell.poles for cell in self.all_cells[self.current_trench][self.i]])
            for cell in trench_cells:
                cell.barcode_to_binary(max_bit=8)
                barcode_list.append(cell.barcode)
                poles_list.append(cell.poles)
            # Todo: deal with cumulative
            if self.tracked:
                self.all_cells[self.current_trench][self.i + 1] = trench_cells
            else:
                self.all_cells[self.current_trench].append(trench_cells)
            data_buffer["label"].append([cell.label for cell in self.all_cells[self.current_trench][self.i + 1]
                                         if self.buffer_next is not None])
            data_buffer["barcode"].append([cell.barcode for cell in self.all_cells[self.current_trench][self.i+1]])
            data_buffer["poles"].append([cell.poles for cell in self.all_cells[self.current_trench][self.i+1]])

            # print(np.mean(number_cells))

            if update_para:
                self.update_model_para("lineage-trench")
            if ret_df:
                trench_track = [data_buffer["trench"]] * len(data_buffer["track_frame"])
                track_df = pd.DataFrame(data={
                    "trench_id": trench_track,
                    "time_(mins)": data_buffer["track_frame"],
                    "label": data_buffer["label"],
                    "parent_label-1": data_buffer["track-1"],
                    "confidence-1": data_buffer["confidence-1"],
                    "parent_label-2": data_buffer["track-2"],
                    "confidence-2": data_buffer["confidence-2"],
                    "centroid": data_buffer["coord"],
                    "barcode": data_buffer["barcode"],
                    "poles": data_buffer["poles"]
                })
                trench_lyse = [data_buffer["trench"]] * len(data_buffer["lysis_frame"])
                lysis_df = pd.DataFrame(data={
                    "trench_id": trench_lyse,
                    "time_(mins)": data_buffer["lysis_frame"],
                    "label-1": data_buffer["lysis-1"]
                    # "confidence-1": data_buffer["confidence-1"]
                    # "label-2": data_buffer["lysis-2"],
                    # "confidence-2": data_buffer["confidence-2"]
                })
                return track_df, lysis_df
            return data_buffer
        else:
            return f"""Specified trench id {trench} does not exist"""

    def track_trench_iteratively(self, trench, threshold=-1, max_dpf=2, search_mode="SeqMatch", p_sp=-1,
                                 special_reporter=None, show_details=False, fill_gap=False,
                                 adap_dpf=True, drift=False, skew_model=True, update_para=True, thresh_per_iter=200, radius=0):
        sys.setrecursionlimit(999999999) # may need to losen the limit
        if threshold == -1:
            threshold = self.max_y
        no_steps = round(threshold / thresh_per_iter)
        # print(no_steps)
        self.update_model_para("unif")
        probability_mode = "sizer-adder"
        for i in range(no_steps - 1):
            thr = int(threshold * (i + 1) / no_steps)
            _ = self.track_trench(trench=trench, threshold=thr, max_dpf=max_dpf, search_mode=search_mode, probability_mode=probability_mode, 
                              p_sp=p_sp, special_reporter=special_reporter, show_details=show_details,
                              ret_df=False, fill_gap=fill_gap, adap_dpf=adap_dpf, drift=drift, skew_model=skew_model, 
                              update_para=update_para, cumulative=True, radius=radius)
            if self.sizer_length_paras[1] == 0 or self.adder_length_paras[1] == 0:
                self.update_model_para("unif")
            else:
                probability_mode = "sizer-adder"
        return (self.track_trench(trench=trench, threshold=threshold, max_dpf=max_dpf, search_mode=search_mode, 
                                  probability_mode=probability_mode, p_sp=p_sp, special_reporter=special_reporter,
                                  show_details=show_details, ret_df=False, fill_gap=fill_gap, adap_dpf=adap_dpf, 
                                  drift=drift, skew_model=skew_model, update_para=update_para,
                                  cumulative=True, radius=radius),
                self.all_cells)

    def track_trenches_iteratively(self, trenches=None, threshold=-1, max_dpf=2, search_mode="SeqMatch", p_sp=-1,
                                   special_reporter=None, show_details=False, save_dir="./temp/", ret_df=False,
                                   fill_gap=False, adap_dpf=True, drift=False, skew_model=True, update_para=True, 
                                   thresh_per_iter=200, radius=0):
        if trenches is None:
            trenches = self.trenches
        results = Parallel(n_jobs=-1, verbose=5)(delayed(self.track_trench_iteratively)
                                                 (trench=t, threshold=threshold, max_dpf=max_dpf, search_mode=search_mode, 
                                                  p_sp=p_sp, special_reporter=special_reporter, show_details=show_details, 
                                                  fill_gap=fill_gap, adap_dpf=adap_dpf, drift=drift, skew_model=skew_model, 
                                                  update_para=update_para, thresh_per_iter=thresh_per_iter, radius=radius)
                                                 for t in trenches)
        data_buffer = {
            "trench_track": [],
            "trench_lyse": [],
            "track-1": [],
            "lysis-1": [],
            "confidence-1": [],
            "track-2": [],
            "lysis-2": [],
            "confidence-2": [],
            "label": [],
            "lysis_frame": [],
            "track_frame": [],
            "coord": [],
            "barcode": [],
            "poles": []
        }
        for r in results:
            data_buffer["trench_track"].extend([r[0]["trench"]] * len(r[0]["track_frame"]))
            data_buffer["trench_lyse"].extend([r[0]["trench"]] * len(r[0]["lysis_frame"]))
            data_buffer["track-1"].extend(r[0]["track-1"])
            data_buffer["lysis-1"].extend(r[0]["lysis-1"])
            data_buffer["confidence-1"].extend(r[0]["confidence-1"])
            data_buffer["track-2"].extend(r[0]["track-2"])
            data_buffer["lysis-2"].extend(r[0]["lysis-2"])
            data_buffer["confidence-2"].extend(r[0]["confidence-2"])
            data_buffer["label"].extend(r[0]["label"])
            data_buffer["lysis_frame"].extend(r[0]["lysis_frame"])
            data_buffer["track_frame"].extend(r[0]["track_frame"])
            data_buffer["coord"].extend(r[0]["coord"])
            data_buffer["barcode"].extend(r[0]["barcode"])
            data_buffer["poles"].extend(r[0]["poles"])
            # k = list(r[1].keys())[0]
            k = r[0]["trench"]
            self.all_cells[k] = r[1][k]
        track_df = pd.DataFrame(data={
            "trench_id": data_buffer["trench_track"],
            "time_(mins)": data_buffer["track_frame"],
            "label": data_buffer["label"],
            "parent_label-1": data_buffer["track-1"],
            "confidence-1": data_buffer["confidence-1"],
            "parent_label-2": data_buffer["track-2"],
            "confidence-2": data_buffer["confidence-2"],
            "centroid": data_buffer["coord"],
            "barcode": data_buffer["barcode"],
            "poles": data_buffer["poles"]
        })
        lysis_df = pd.DataFrame(data={
            "trench_id": data_buffer["trench_lyse"],
            "time_(mins)": data_buffer["lysis_frame"],
            "label-1": data_buffer["lysis-1"]
            # "confidence-1": data_buffer["confidence-1"]
            # "label-2": data_buffer["lysis-2"],
            # "confidence-2": data_buffer["confidence-2"]
        })
        if ret_df:
            return track_df, lysis_df
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        file_track = "track_TR"
        file_lyse = "lysis_TR"
        file_cell = "cells_TR"
        for t in trenches:
            file_track += "_" + str(int(t))
            file_lyse += "_" + str(int(t))
            file_cell += "_" + str(int(t))
        file_track = os.path.join(save_dir, file_track) + ".csv"
        file_lyse = os.path.join(save_dir, file_lyse) + ".csv"
        track_df.to_csv(file_track)
        lysis_df.to_csv(file_lyse)
        # Todo: save the list of Cell objects to files using `pickle`
        # pickle.dump(self.all_cells, os.path.join(save_dir, file_cell))
        print(self.all_cells.keys())
        return f"""output saved at {file_track} and {file_lyse}."""

    def measure_div_dist(self, trenches, plot=False):
        div_dist = []
        freq = {}
        if isinstance(trenches, list):
            if plot:
                subplots = len(trenches)
                cols = 2
                rows = round(np.ceil(subplots / cols))
                fig, axes = plt.subplots(nrows=rows, ncols=cols, dpi=80, figsize=(20, 20))
                axes_flat = axes.flatten()
                i = 0
            for tr in trenches:
                for frame in self.all_cells[tr]:
                    div_dist.append(len([cell for cell in frame if cell.divide is True]))
                if plot:
                    i += 1
                    axes_flat[i].plot(range(len(self.all_cells[tr])),
                                      div_dist[-len(self.all_cells[tr]):])
                    axes_flat[i].set_title(f"trench {tr}")
        if isinstance(trenches, int):
            for frame in self.all_cells[trenches]:
                div_dist.append(len([cell for cell in frame if cell.divide is True]))
            if plot:
                plt.figure(figsize=(10, 10))
                plt.plot(range(len(self.all_cells[trenches])),
                         div_dist)
                plt.title(f"trench {trenches}")
                plt.show()
        for i in div_dist:
            if i in freq:
                freq[i] += 1
            else:
                freq[i] = 1
        for i in freq:
            freq[i] = freq[i] / len(div_dist)
        return freq

    def generate_lineage(self, trench, mode="full", frame=None, label=None):
        if mode == "full":
            return Lineage.from_tracker(self.all_cells[trench][0])
        elif mode == "footprint":
            cell = self.all_cells[trench][frame][label - 1]
            line = [cell]
            while cell.parent is not None:
                cell = cell.parent
                line.insert(0, cell)
            return Lineage(line)
        elif mode == "offspring":
            mother_cell = self.all_cells[trench][frame][label - 1]
            return Lineage.from_tracker([mother_cell])


# from Charlie
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


# https://stackoverflow.com/questions/50592576/generate-unique-binary-permutations-in-python
def binary_permutations(lst):
    for com in itertools.combinations(range(len(lst)), lst.count(1)):
        result = [0] * len(lst)
        for i in com:
            result[i] = 1
        yield result


# Azzalini and Capitanio (2014)
def sn_cdf(val, a, mean, var):
    # Todo: add upper / lower tail handling

    val = (np.array(val) - mean) / np.sqrt(var)
    if a >= 0:
        x = np.array([[v, 0] for v in val])
        rho = -a / np.sqrt(1 + a ** 2)
        cov = np.array([[1, rho], [rho, 1]])
        return 2 * multivariate_normal.cdf(x, cov=cov)
    else:
        x = np.array([[-v, 0] for v in val])
        rho = a / np.sqrt(1 + a ** 2)
        cov = np.array([[1, rho], [rho, 1]])
        return 1 - 2 * multivariate_normal.cdf(x, cov=cov)
    # if a >= 0:
    #     # x = np.array([[v, 0] for v in val])
    #     # rho = -a / np.sqrt(1 + a ** 2)
    #     # cov = np.array([[1, rho], [rho, 1]])
    #     # return 2 * multivariate_normal.cdf(x, cov=cov)
    #     return norm.cdf(val) - 2 * owens_t(val, np.array([a] * len(val)))
    # else:
    #     # x = np.array([[-v, 0] for v in val])
    #     # rho = a / np.sqrt(1 + a ** 2)
    #     # cov = np.array([[1, rho], [rho, 1]])
    #     # return 1 - 2 * multivariate_normal.cdf(x, cov=cov)
    #     return 1 - (norm.cdf(-val) - 2 * owens_t(-val, np.array([-a] * len(val))))


def scoring(p, d):
    scoring_mode = "posterior"
    if scoring_mode == "weight-d":
        if p > 0:
            return (0.75 + 1 / (1 - np.log10(p))) / ((np.mean(d)) ** 3 + 1)
        elif p == 0:
            return 0.75 / ((np.mean(d)) ** 3 + 1)
        else:
            return 0
    elif scoring_mode == "posterior":
        if 1 >= p >= 0:
            # return (4 + p) * (np.exp(1 / np.mean(d)) - 1)
            return (3 + p) * (np.exp(1 / np.mean(d)) - 1)
        else:
            return 0
