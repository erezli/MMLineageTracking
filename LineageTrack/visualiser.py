import os
import cv2 as cv
import pandas as pd
import numpy as np

template_mask = ['xy', '_mCherry_TR', '_T', '-_epoch-20_prob-99.png']

def generate_file_name(template, pre, FOV, trench, time):
    # e.g., xy000_mCherry_TR1_T0000-_epoch-20_prob-99.png
    path = str(pre) + template[0] + str(FOV) + template[1] + str(int(trench)) + template[2] + str(time) + template[3]
    return path


class Visualiser:
    def __init__(self, FOV, track_df, lysis_df, cells=None):
        self.FOV = FOV
        track_df.sort_values(["trench_id", "time_(mins)"], inplace=True)
        lysis_df.sort_values(["trench_id", "time_(mins)"], inplace=True)
        self.track_df = track_df
        self.lysis_df = lysis_df
        self.cells = cells
        self.trenches = sorted(list(set(self.track_df.loc[:, "trench_id"])))

    @classmethod
    def from_path(cls, FOV, filepath_t, filepath_l):
        if not os.path.exists(filepath_t):
            raise Exception(f"specified file {filepath_t} not found")
        df_t = pd.read_csv(filepath_t)
        if not os.path.exists(filepath_l):
            raise Exception(f"specified file {filepath_l} not found")
        df_l = pd.read_csv(filepath_l)
        return cls(FOV, df_t, df_l)

    def label_images(self, mask_dir, mode="connect_daughter", save_dir="./temp/labelled_masks/", template=template_mask):
        """

        @param save_dir:
        @param mask_dir:
        @param mode: connect_daughter; binary
        @param template:
        @return:
        """
        if template is None:
            template = template_mask
        for t in self.trenches:
            times = self.track_df.loc[self.track_df["trench_id"] == t, "time_(mins)"].copy()
            times = sorted(list(set(times)))
            if mode == "connect_daughter":
                for i in range(len(times)):
                    time = times[i]
                    frame = "%04d" % i
                    cells = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                              (self.track_df["time_(mins)"] == time)].copy()
                    cells.reset_index(drop=True, inplace=True)
                    path1 = generate_file_name(template, "", self.FOV, t, frame)
                    mask = cv.imread(mask_dir + os.path.sep + path1)
                    # print(mask.shape)
                    for c in range(len(cells.at[0, "label"])):
                        position = (round(cells.at[0, "centroid"][c][0]), round(cells.at[0, "centroid"][c][1]))
                        cv.drawMarker(mask, position, (255, 0, 0))
                        if c > 0:
                            if (cells.at[0, "parent_label"][c] == cells.at[0, "parent_label"][c - 1]) & \
                                    (cells.at[0, "parent_label"][c] is not None):
                                position_2 = (round(cells.at[0, "centroid"][c - 1][0]),
                                              round(cells.at[0, "centroid"][c - 1][1]))
                                cv.line(mask, position, position_2, (0, 255, 0), thickness=2)
                    path2 = generate_file_name(template, "labelled_", self.FOV, t, frame)
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    cv.imwrite(save_dir + os.path.sep + path2, mask)
