import os
import cv2 as cv
import pandas as pd
import numpy as np
import ast

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
        df_t = pd.read_csv(filepath_t, converters={
            "label": ast.literal_eval,
            "parent_label": ast.literal_eval,
            "centroid": ast.literal_eval,
            "barcode": ast.literal_eval,
            "poles": ast.literal_eval
        })
        if not os.path.exists(filepath_l):
            raise Exception(f"specified file {filepath_l} not found")
        df_l = pd.read_csv(filepath_l, converters={"label": ast.literal_eval})
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
            # for Python 3.10
            # match mode:
            #     case "connect_daughter"
            if mode == "connect_daughter":
                for i in range(len(times)):
                    time = times[i]
                    frame = "%04d" % i
                    cells = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                              (self.track_df["time_(mins)"] == time)].copy()
                    cells.reset_index(drop=True, inplace=True)
                    read_path = generate_file_name(template, "", self.FOV, t, frame)
                    mask = cv.imread(mask_dir + os.path.sep + read_path)
                    # print(mask.shape)
                    for c in range(len(cells.at[0, "label"])):
                        position_1 = (round(cells.at[0, "centroid"][c][0]), round(cells.at[0, "centroid"][c][1]))
                        cv.drawMarker(mask, position_1, (255, 0, 0))
                        if c > 0:
                            if (cells.at[0, "parent_label"][c] == cells.at[0, "parent_label"][c - 1]) & \
                                    (cells.at[0, "parent_label"][c] is not None):
                                position_2 = (round(cells.at[0, "centroid"][c - 1][0]),
                                              round(cells.at[0, "centroid"][c - 1][1]))
                                cv.line(mask, position_1, position_2, (0, 255, 0), thickness=2)
                                middle_position = (round((position_1[0] + position_2[0])/2) + 5,
                                                   round((position_1[1] + position_2[1])/2) - 7)
                                cv.putText(mask, "divide",
                                           middle_position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                                middle_position = (middle_position[0], middle_position[1] + 7)
                                cv.putText(mask, "from",
                                           middle_position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                                middle_position = (middle_position[0], middle_position[1] + 7)
                                cv.putText(mask, f"cell no{int(cells.at[0, 'parent_label'][c])}",
                                           middle_position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                    write_path = generate_file_name(template, "labelled_", self.FOV, t, frame)
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    cv.imwrite(save_dir + os.path.sep + write_path, mask)
            elif mode == "landscape-line":
                offset = 0
                landscape = None
                mask_buffer = None
                for i in range(len(times) - 1):
                    time1 = times[i]
                    time2 = times[i + 1]
                    frame1 = "%04d" % i
                    frame2 = "%04d" % (i + 1)
                    cells1 = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                              (self.track_df["time_(mins)"] == time1)].copy()
                    cells1.reset_index(drop=True, inplace=True)
                    cells2 = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                               (self.track_df["time_(mins)"] == time2)].copy()
                    cells2.reset_index(drop=True, inplace=True)

                    path2 = generate_file_name(template, "", self.FOV, t, frame2)
                    mask2 = cv.imread(mask_dir + os.path.sep + path2)
                    cv.putText(mask2, "t={}".format(time2),
                               (10, 15), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                    if i == 0:
                        path1 = generate_file_name(template, "", self.FOV, t, frame1)
                        mask1 = cv.imread(mask_dir + os.path.sep + path1)
                        cv.putText(mask1, "t={}".format(time1),
                                   (10, 15), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                        landscape = mask1
                    else:
                        mask1 = mask_buffer
                    landscape = np.concatenate((landscape, mask2), axis=1)
                    mask_buffer = mask2
                    for c in range(len(cells2.at[0, "label"])):
                        if cells2.at[0, "parent_label"][c] is not None:
                            parent = int(cells2.at[0, "parent_label"][c]) - 1
                            position1 = (round(cells2.at[0, "centroid"][c][0] + offset + mask1.shape[1]),
                                         round(cells2.at[0, "centroid"][c][1]))
                            position2 = (round(cells1.at[0, "centroid"][parent][0] + offset),
                                         round(cells1.at[0, "centroid"][parent][1]))
                            cv.line(landscape, position1, position2, (0, 255, 0), thickness=2)
                    offset += mask1.shape[1]
                write_path = generate_file_name(template, "landscape_line_", self.FOV, t, "")
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                cv.imwrite(save_dir + os.path.sep + write_path, landscape)
                print(f"saved as {save_dir + os.path.sep + write_path}")
            elif mode == "barcode":
                for i in range(len(times)):
                    time = times[i]
                    frame = "%04d" % i
                    cells = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                              (self.track_df["time_(mins)"] == time)].copy()
                    cells.reset_index(drop=True, inplace=True)
                    read_path = generate_file_name(template, "", self.FOV, t, frame)
                    mask = cv.imread(mask_dir + os.path.sep + read_path)
                    for c in range(len(cells.at[0, "label"])):
                        if cells.at[0, "barcode"][c] is not None:
                            position = (round(cells.at[0, "centroid"][c][0]), round(cells.at[0, "centroid"][c][1]))
                            cv.putText(mask, "{:08b}".format(int(cells.at[0, "barcode"][c], 2)),
                                       position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                    write_path = generate_file_name(template, "barcoded_", self.FOV, t, frame)
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    cv.imwrite(save_dir + os.path.sep + write_path, mask)
            elif mode == "landscape-gray-scale":
                landscape = None
                for i in range(len(times) - 1):
                    if i == 0:
                        time1 = times[i]
                        frame1 = "%04d" % i
                        cells1 = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                                   (self.track_df["time_(mins)"] == time1)].copy()
                        cells1.reset_index(drop=True, inplace=True)
                        path1 = generate_file_name(template, "", self.FOV, t, frame1)
                        mask1 = cv.imread(mask_dir + os.path.sep + path1, cv.IMREAD_GRAYSCALE)
                        contours, hierarchy = cv.findContours(mask1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                        rev_con = ()
                        for k in reversed(contours):
                            rev_con = rev_con + (k,)
                        cv.putText(mask1, "t={}".format(time1),
                                   (10, 15), cv.FONT_HERSHEY_COMPLEX, 0.5, 180)
                        max_gray = int("11111111", 2)
                        mask1 = cv.cvtColor(mask1, cv.COLOR_GRAY2RGB)
                        for c in range(len(cells1.at[0, "label"])):
                            barcode = cells1.at[0, "barcode"][c]
                            if barcode is not None:
                                gray_scale = 200 - (int(barcode, 2) / max_gray) * 200
                                cv.drawContours(mask1, rev_con, contourIdx=c,
                                                color=(int(gray_scale), int(gray_scale), int(gray_scale)), thickness=-1)
                                position = (round(cells1.at[0, "centroid"][c][0]), round(cells1.at[0, "centroid"][c][1]))
                                cv.putText(mask1, "{:08b}".format(int(cells1.at[0, "barcode"][c], 2)),
                                           position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (180, 180, 180))
                        landscape = mask1
                    time2 = times[i + 1]
                    frame2 = "%04d" % (i + 1)
                    cells2 = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                               (self.track_df["time_(mins)"] == time2)].copy()
                    cells2.reset_index(drop=True, inplace=True)
                    path2 = generate_file_name(template, "", self.FOV, t, frame2)
                    mask2 = cv.imread(mask_dir + os.path.sep + path2, cv.IMREAD_GRAYSCALE)
                    contours, hierarchy = cv.findContours(mask2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                    rev_con = ()
                    for k in reversed(contours):
                        rev_con = rev_con + (k,)
                    cv.putText(mask2, "t={}".format(time2),
                               (10, 15), cv.FONT_HERSHEY_COMPLEX, 0.5, 180)
                    max_gray = int("11111111", 2)
                    mask2 = cv.cvtColor(mask2, cv.COLOR_GRAY2RGB)
                    for c in range(len(cells2.at[0, "label"])):
                        barcode = cells2.at[0, "barcode"][c]
                        if barcode is not None:
                            gray_scale = 200 - (int(barcode, 2) / max_gray) * 200
                            cv.drawContours(mask2, rev_con, contourIdx=c,
                                            color=(int(gray_scale), int(gray_scale), int(gray_scale)), thickness=-1)
                            position = (round(cells2.at[0, "centroid"][c][0]), round(cells2.at[0, "centroid"][c][1]))
                            cv.putText(mask2, "{:08b}".format(int(cells2.at[0, "barcode"][c], 2)),
                                       position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (180, 180, 180))
                    landscape = np.concatenate((landscape, mask2), axis=1)
                write_path = generate_file_name(template, "landscape_gray_", self.FOV, t, "")
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                cv.imwrite(save_dir + os.path.sep + write_path, landscape)
                print(f"saved as {save_dir + os.path.sep + write_path}")
            elif mode == "generation-by-poles":
                for i in range(len(times)):
                    time = times[i]
                    frame = "%04d" % i
                    cells = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                              (self.track_df["time_(mins)"] == time)].copy()
                    cells.reset_index(drop=True, inplace=True)
                    read_path = generate_file_name(template, "", self.FOV, t, frame)
                    mask = cv.imread(mask_dir + os.path.sep + read_path)
                    for c in range(len(cells.at[0, "label"])):
                        if cells.at[0, "poles"][c] is not None:
                            position = (round(cells.at[0, "centroid"][c][0]), round(cells.at[0, "centroid"][c][1]))
                            cv.putText(mask, "{}".format(cells.at[0, "poles"][c]),
                                       position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                    write_path = generate_file_name(template, "poles_", self.FOV, t, frame)
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    cv.imwrite(save_dir + os.path.sep + write_path, mask)
            else:
                return "mode not correct"