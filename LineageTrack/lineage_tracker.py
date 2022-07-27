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
    pointer = 0
    for count, row in enumerate(formatted_pixel_lists):
        # new_row = [int(x) for x in row if x != ""]
        new_row = []
        for x in row:
            if x != "":
                try:
                    new_row.append(int(x))
                except ValueError:
                    null_data_counter += 1
                    print(row)
                    new_row.append(0)
        formatted_pixel_lists[count] = new_row
    if null_data_counter != 0:
        print(null_data_counter)
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
            print("error")
        for path in args:
            if not os.path.exists(path):
                print("error")
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
                              "centroid_local-0", "centroid_local-1", "orientation"]]
        for d in df_list:
            channel = d.loc[1, "channel"]
            self.channels.append(channel)
            self.df.loc[:, "{}_intensity_mean".format(channel)] = d.loc[:, "intensity_mean"]
            # self.df.loc[:, "{}_image_intensity".format(channel)] = d.loc[:, "image_intensity"]
            # might include later but takes a lot of memory
            self.df.loc[:, "{}_identity".format(channel)] = d.loc[:, "identity"]
        self.channels = sorted(list(set(self.channels)))
        self.properties = sorted(list(set(self.df.columns)))
        self.df.sort_values(["trench_id", "time_(mins)", "label"], inplace=True)

        print(self.channels)
        print(self.df.columns)
        # print(self.df.head(1))

    def __str__(self):
        return f"""
            Read {len(self.files)} files
            Channels: {self.channels}
            Properties for each cell: {self.properties}
        """

    def load_trench_frame(self, trench_id, time):
        points = np.array()
        return points