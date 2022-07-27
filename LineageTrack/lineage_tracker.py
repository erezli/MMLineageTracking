import os
from glob import glob
import re
import pandas as pd
import numpy as np
from scipy.spatial import KDTree


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
        #print(null_data_counter)
        print("Some data is lost due to compression in image intensity column")
    pixel_array = np.array(formatted_pixel_lists)

    return pixel_array


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
        print(self.files)

        # df_list = [pd.read_csv(f, converters = {"image_intensity":reconstruct_array_from_str}) for f in self.files]
        df_list = [pd.read_csv(f, dtype={"major_axis_length": np.float32, "minor_axis_length": np.float32,
                                         "centroid-0": np.float32, "centroid-1": np.float32,
                                         "centroid_local-0": np.float32, "centroid_local-1": np.float32,
                                         "orientation": np.float32, "intensity_mean": np.float32},
                               converters={"image_intensity": reconstruct_array_from_str}) for f in self.files]
        # Todo: use Zarr array to reduce memory usage
        self.channels = []
        self.df = df_list[0][["trench_id", "time_(mins)", "label", "area", "major_axis_length",
                              "minor_axis_length", "centroid-0", "centroid-1",
                              "centroid_local-0", "centroid_local-1", "orientation"]].copy()
        for d in df_list:
            channel = d.loc[1, "channel"]
            self.channels.append(channel)
            # self.df.loc[:, "{}_intensity_mean".format(channel)] = d.loc[:, "intensity_mean"]
            self.df.insert(len(self.df.columns), "{}_intensity_mean".format(channel), d.loc[:, "intensity_mean"])
            # self.df.insert(len(self.df.columns),"{}_image_intensity".format(channel), d.loc[:, "image_intensity"])
            # might include later but takes a lot of memory
            self.df.insert(len(self.df.columns), "{}_identity".format(channel), d.loc[:, "identity"])
        self.channels = sorted(list(set(self.channels)))
        self.properties = sorted(list(set(self.df.columns)))
        self.df.sort_values(["trench_id", "time_(mins)", "label"], inplace=True)
        self.current_trench = 0
        self.current_frame = 0
        self.frames = []

        print(self.channels)
        print(self.df.columns)
        print(self.df.head(1))

    def __str__(self):
        return f"""
            Read {len(self.files)} files
            Channels: {self.channels}
            Properties for each cell: {self.properties}
        """

    def load_trench_frame(self, trench_id, time):
        points = np.array()
        return points

    def untitled(self):
        trench_list = self.df.col[:, "trench_id"]
        trench_list = sorted(list(set(trench_list)))
        for tr in trench_list:
            self.current_trench = tr
            self.frames = self.df.col[self.df["trench_id"] == self.current_trench, ["time_(mins)"]]
            self.frames = sorted(list(set(self.frames)))
            for f in self.frames:
                self.current_frame = f
                # load all cells in this frame and this trench
