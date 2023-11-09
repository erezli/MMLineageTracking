import os
import cv2 as cv
import pandas as pd
import numpy as np
import ast
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zarr
from tqdm import tqdm


# load from dataframes, visualise using label_images(zarr_dir, channel, mode="landscape-line", save_dir="./temp/")
class Visualiser:
    def __init__(self, track_df, lysis_df, cells=None):
        track_df.sort_values(["trench_id", "time_(mins)"], inplace=True)
        lysis_df.sort_values(["trench_id", "time_(mins)"], inplace=True)
        self.track_df = track_df
        self.lysis_df = lysis_df
        self.cells = cells
        self.trenches = sorted(list(set(self.track_df.loc[:, "trench_id"])))
        flat_list_y = [item[1] for sublist in self.track_df.loc[:, "centroid"] for item in sublist]
        self.max_y = max(flat_list_y)
        self.image_width = None

    @classmethod
    def from_path(cls, filepath_t, filepath_l):
        if not os.path.exists(filepath_t):
            raise Exception(f"specified file {filepath_t} not found")
        df_t = pd.read_csv(filepath_t, converters={
            "label": ast.literal_eval,
            "parent_label-1": ast.literal_eval,
            "centroid": ast.literal_eval,
            "barcode": ast.literal_eval,
            "poles": ast.literal_eval
        })
        if not os.path.exists(filepath_l):
            raise Exception(f"specified file {filepath_l} not found")
        df_l = pd.read_csv(filepath_l, converters={"label": ast.literal_eval})
        return cls(df_t, df_l)

    def label_images(self, zarr_dir, channel, mode="connect_daughter", save_dir="./temp/labelled_masks/",
                     show_other=True, for_frames=None, colour_scale="Greys", fluores=False, step=1, skip=0):
        """
        Generate images that is labelled by specific mode
        @param zarr_dir: directory of the images to label
        @param channel: index for the channel you want to visualise
        @param mode: connect_daughter; landscape-line
        @param save_dir: directory to save the labelled images
        @param show_other: whether to show the un-tracked cells or not (in barcode mode)
        @param for_frames: in some landscape mode, can set this to a 2 element tuple
        to specify the range of frames to label
        @param colour_scale: 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        @param fluores: integer to scale the intensity of fluorescence images
        @param step: time step in going through the images
        @param skip: skip initial frames
        @return:
        """
        for t in self.trenches:
            times = self.track_df.loc[self.track_df["trench_id"] == t, "time_(mins)"].copy()
            times = sorted(list(set(times)))
            self.times = times
            # for Python 3.10
            # match mode:
            #     case "connect_daughter"
            if mode == "connect_daughter":
                for i in range(len(times)):
                    time = times[i]
                    frame = "%04d" % ((i * step) + skip)
                    cells = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                              (self.track_df["time_(mins)"] == time)].copy()
                    cells.reset_index(drop=True, inplace=True)
                    # read_path = generate_file_name(template, "", self.FOV, t, frame, mode=template_mode)
                    image = zarr.open(zarr_dir, mode='r')[t, i, channel, :, :]  # cv.imread(image_dir + os.path.sep + read_path)
                    image = np.asarray(image)
                    for c in range(len(cells.at[0, "label"])):
                        position_1 = (round(cells.at[0, "centroid"][c][0]), round(cells.at[0, "centroid"][c][1]))
                        cv.drawMarker(image, position_1, (255, 0, 0))
                        if c > 0:
                            if (cells.at[0, "parent_label-1"][c] == cells.at[0, "parent_label-1"][c - 1]) & \
                                    (cells.at[0, "parent_label-1"][c] is not None):
                                position_2 = (round(cells.at[0, "centroid"][c - 1][0]),
                                              round(cells.at[0, "centroid"][c - 1][1]))
                                cv.line(image, position_1, position_2, (0, 255, 0), thickness=2)
                                middle_position = (round((position_1[0] + position_2[0])/2) + 5,
                                                   round((position_1[1] + position_2[1])/2) - 7)
                                cv.putText(image, "divide",
                                           middle_position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                                middle_position = (middle_position[0], middle_position[1] + 7)
                                cv.putText(image, "from",
                                           middle_position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                                middle_position = (middle_position[0], middle_position[1] + 7)
                                cv.putText(image, f"cell no{int(cells.at[0, 'parent_label-1'][c])}",
                                           middle_position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                    write_path = "connected_TR{}_C{}.png".format(t, channel)
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    cv.imwrite(save_dir + os.path.sep + write_path, image)

            elif mode == "landscape-line":
                offset = 0
                landscape = None
                image_buffer = None
                if for_frames:
                    idx = range(for_frames[0], for_frames[1])
                else:
                    idx = range(len(times) - 1)
                for i in tqdm(idx, desc=f"trench {t}"):
                    time1 = times[i]
                    time2 = times[i + 1]
                    frame1 = "%04d" % ((i * step) + skip)
                    frame2 = "%04d" % (((i + 1) * step) + skip)
                    cells1 = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                               (self.track_df["time_(mins)"] == time1)].copy()
                    cells1.reset_index(drop=True, inplace=True)
                    cells2 = self.track_df.loc[(self.track_df["trench_id"] == t) &
                                               (self.track_df["time_(mins)"] == time2)].copy()
                    cells2.reset_index(drop=True, inplace=True)

                    image2 = zarr.open(zarr_dir, mode='r')[t, i + 1, channel, :, :]
                    # image2 = np.asarray(image2)
                    image2 = cv.cvtColor(image2, cv.COLOR_GRAY2RGB)
                    if isinstance(fluores, int) and fluores != 0:
                        image2 *= fluores
                    cv.putText(image2, "t={}".format(time2),
                               (0, 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                    cv.putText(image2, "Conf={:.1f}%".format(cells2.at[0, "confidence-1"] * 100),
                               (0, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                    if i == 0:
                        image1 = zarr.open(zarr_dir, mode='r')[t, i, channel, :, :]
                        # image1 = np.asarray(image1)
                        image1 = cv.cvtColor(image1, cv.COLOR_GRAY2RGB)
                        if isinstance(fluores, int) and fluores != 0:
                            image1 *= fluores
                        self.image_width = image1.shape[1]
                        cv.putText(image1, "t={}".format(time1),
                                   (0, 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
                        landscape = image1
                    else:
                        image1 = image_buffer
                    landscape = np.concatenate((landscape, image2), axis=1)
                    image_buffer = image2
                    for c in range(len(cells2.at[0, "label"])):
                        if cells2.at[0, "parent_label-1"][c] is not None:
                            parent = int(cells2.at[0, "parent_label-1"][c]) - 1
                            position1 = (round(cells2.at[0, "centroid"][c][0] + offset + image1.shape[1]),
                                         round(cells2.at[0, "centroid"][c][1]))
                            position2 = (round(cells1.at[0, "centroid"][parent][0] + offset),
                                         round(cells1.at[0, "centroid"][parent][1]))
                            cv.line(landscape, position1, position2, (0, 255, 0), thickness=2)
                    offset += image1.shape[1]
                write_path = "landscape_line_TR{}_C{}.png".format(t, channel)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                cv.imwrite(save_dir + os.path.sep + write_path, landscape)
                print(f"saved as {save_dir + os.path.sep + write_path}")

            # elif mode == "barcode":
            #     for i in range(len(times)):
            #         time = times[i]
            #         frame = "%04d" % ((i * step) + skip)
            #         cells = self.track_df.loc[(self.track_df["trench_id"] == t) &
            #                                   (self.track_df["time_(mins)"] == time)].copy()
            #         cells.reset_index(drop=True, inplace=True)
            #         read_path = generate_file_name(template, "", self.FOV, t, frame, mode=template_mode)
            #         image = cv.imread(image_dir + os.path.sep + read_path, cv.IMREAD_GRAYSCALE)
            #         contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            #         rev_con = ()
            #         for k in reversed(contours):
            #             rev_con = rev_con + (k,)
            #         image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            #         flat_list_barcode = [int(item, 2)
            #                              for sublist in self.track_df.loc[:, "barcode"]
            #                              for item in sublist if item is not None]
            #         flat_list_barcode = sorted(list(set(flat_list_barcode)))
            #         cmap = matplotlib.cm.get_cmap(colour_scale)
            #         # max_gray = max(flat_list_barcode)
            #         for c in range(len(cells.at[0, "label"])):
            #             if cells.at[0, "barcode"][c] is not None:
            #                 position = (round(cells.at[0, "centroid"][c][0]), round(cells.at[0, "centroid"][c][1]))
            #                 cv.putText(image, "{:08b}".format(int(cells.at[0, "barcode"][c], 2)),
            #                            position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (180, 180, 180))
            #                 # gray_scale = 250 - (int(cells.at[0, "barcode"][c], 2) / max_gray) * 200
            #                 # gray_scale = 250 - (flat_list_barcode.index(int(cells.at[0, "barcode"][c], 2)) /
            #                 #                     len(flat_list_barcode)) * 250
            #                 color_scale = cmap((flat_list_barcode.index(int(cells.at[0, "barcode"][c], 2)) /
            #                                     len(flat_list_barcode)))
            #                 color_scale = tuple(reversed([int((255 * x)) for x in color_scale[0:3]]))
            #                 cv.drawContours(image, rev_con, contourIdx=c,
            #                                 color=color_scale,
            #                                 thickness=-1)
            #         write_path = generate_file_name(template, "barcoded_", self.FOV, t, frame, mode=template_mode)
            #         if not os.path.isdir(save_dir):
            #             os.mkdir(save_dir)
            #         cv.imwrite(save_dir + os.path.sep + write_path, image)

            # elif mode == "landscape-colour-scale":
            #     if for_frames:
            #         idx = range(for_frames[0], for_frames[1])
            #     else:
            #         idx = range(len(times) - 1)
            #     cells_barcode = self.track_df.loc[(self.track_df["trench_id"] == t) &
            #                                       (self.track_df["time_(mins)"] >= times[idx[0]]) &
            #                                       (self.track_df["time_(mins)"] <= times[idx[-1]]), "barcode"].copy()
            #     flat_list_barcode = [int(item, 2) for sublist in cells_barcode for item in sublist if item is not None]
            #     flat_list_barcode = sorted(list(set(flat_list_barcode)))
            #     # max_gray = max(flat_list_barcode)

            #     def paint_cells(X):
            #         _time = times[X]
            #         _frame = "%04d" % ((X * step) + skip)
            #         _cells = self.track_df.loc[(self.track_df["trench_id"] == t) &
            #                                    (self.track_df["time_(mins)"] == _time)].copy()
            #         _cells.reset_index(drop=True, inplace=True)
            #         _path = generate_file_name(template, "", self.FOV, t, _frame, mode=template_mode)
            #         _image = cv.imread(image_dir + os.path.sep + _path, cv.IMREAD_GRAYSCALE)
            #         _contours, _hierarchy = cv.findContours(_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            #         _rev_con = ()
            #         for K in reversed(_contours):
            #             _rev_con = _rev_con + (K,)
            #         cv.putText(_image, "t={}".format(_time),
            #                    (0, 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 180)
            #         _image = cv.cvtColor(_image, cv.COLOR_GRAY2RGB)
            #         cmap = matplotlib.cm.get_cmap(colour_scale)
            #         for C in range(len(_cells.at[0, "label"])):
            #             _barcode = _cells.at[0, "barcode"][C]
            #             if _barcode is not None:
            #                 # _gray_scale = 250 - (int(_barcode, 2) / max_gray) * 200
            #                 # _gray_scale = 250 - \
            #                 #               (flat_list_barcode.index(int(_barcode, 2)) / len(flat_list_barcode)) * 250
            #                 try:
            #                     _color_scale = cmap(flat_list_barcode.index(int(_barcode, 2)) / len(flat_list_barcode))
            #                     _color_scale = tuple(int((255 * x)) for x in _color_scale[0:3])
            #                     cv.drawContours(_image, _rev_con, contourIdx=C,
            #                                     color=_color_scale,
            #                                     thickness=-1)
            #                 except ValueError:
            #                     print("unseen barcode")
            #                 _position = (round(_cells.at[0, "centroid"][C][0]),
            #                              round(_cells.at[0, "centroid"][C][1]))
            #                 cv.putText(_image, "{:08b}".format(int(_cells.at[0, "barcode"][C], 2)),
            #                            _position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (180, 180, 180))
            #             elif not show_other:
            #                 cv.drawContours(_image, _rev_con, contourIdx=C,
            #                                 color=(0, 0, 0), thickness=-1)
            #         if not show_other:
            #             for x in range(3):
            #                 cv.drawContours(_image, _rev_con, contourIdx=len(_cells.at[0, "label"]) + x,
            #                                 color=(0, 0, 0), thickness=-1)
            #         return _image

            #     landscape = paint_cells(0)
            #     for i in idx:
            #         landscape = np.concatenate((landscape, paint_cells(i+1)), axis=1)
            #     landscape = landscape[:int(self.max_y * 1.05), :]
            #     if for_frames:
            #         time_in_name = str(for_frames[0]) + "-" + str(for_frames[1])
            #     else:
            #         time_in_name = ""
            #     write_path = generate_file_name(template, "landscape_barcode_", self.FOV, t,
            #                                     time_in_name, mode=template_mode)
            #     if not os.path.isdir(save_dir):
            #         os.mkdir(save_dir)
            #     cv.imwrite(save_dir + os.path.sep + write_path, landscape)
            #     print(f"saved as {save_dir + os.path.sep + write_path}")

            # elif mode == "generation-by-poles":
            #     for i in range(len(times)):
            #         time = times[i]
            #         frame = "%04d" % ((i * step) + skip)
            #         cells = self.track_df.loc[(self.track_df["trench_id"] == t) &
            #                                   (self.track_df["time_(mins)"] == time)].copy()
            #         cells.reset_index(drop=True, inplace=True)
            #         read_path = generate_file_name(template, "", self.FOV, t, frame, mode=template_mode)
            #         image = cv.imread(image_dir + os.path.sep + read_path)
            #         for c in range(len(cells.at[0, "label"])):
            #             if cells.at[0, "poles"][c] is not None:
            #                 position = (round(cells.at[0, "centroid"][c][0]), round(cells.at[0, "centroid"][c][1]))
            #                 cv.putText(image, "{}".format(cells.at[0, "poles"][c]),
            #                            position, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
            #         write_path = generate_file_name(template, "poles_", self.FOV, t, frame, mode=template_mode)
            #         if not os.path.isdir(save_dir):
            #             os.mkdir(save_dir)
            #         cv.imwrite(save_dir + os.path.sep + write_path, image)

            else:
                return "mode not correct"

    def highlight_lineage(self, lineages, image_path, save_dir="./temp/labelled_masks/"):
        if isinstance(lineages, list):
            dt = self.times[1] - self.times[0]
            landscape = cv.imread(image_path)
            for line in lineages:
                for i in range(len(line.resident_time) - 1):
                    pt1 = (round(line.positions[i][0] +
                                 (line.resident_time[i] - self.times[0]) / dt * self.image_width),
                           round(line.positions[i][1]))
                    pt2 = (round(line.positions[i+1][0] +
                                 (line.resident_time[i+1] - self.times[0]) / dt * self.image_width),
                           round(line.positions[i+1][1]))
                    cv.line(landscape, pt1, pt2, (255, 0, 0), thickness=3)
            write_path = "highlighted_" + os.path.split(image_path)[1]
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            cv.imwrite(save_dir + os.path.sep + write_path, landscape)
            print(f"saved as {save_dir + os.path.sep + write_path}")

    # def show_second_classification(self, trench, frame, image_dir, save_dir=False, template=None,
    #                                template_mode="exp", fluores=False):

    #     if template is None:
    #         template = template_mask
    #     times = self.track_df.loc[self.track_df["trench_id"] == trench, "time_(mins)"].copy()
    #     times = sorted(list(set(times)))
    #     offset = 0
    #     time1 = times[frame - 1]
    #     time2 = times[frame]
    #     frame1 = "%04d" % (frame - 1)
    #     frame2 = "%04d" % frame
    #     cells1 = self.track_df.loc[(self.track_df["trench_id"] == trench) &
    #                                (self.track_df["time_(mins)"] == time1)].copy()
    #     cells1.reset_index(drop=True, inplace=True)
    #     cells2 = self.track_df.loc[(self.track_df["trench_id"] == trench) &
    #                                (self.track_df["time_(mins)"] == time2)].copy()
    #     cells2.reset_index(drop=True, inplace=True)

    #     path2 = generate_file_name(template, "", self.FOV, trench, frame2, mode=template_mode)
    #     image2 = cv.imread(image_dir + os.path.sep + path2)
    #     if isinstance(fluores, int) and fluores != 0:
    #         image2 *= fluores
    #     cv.putText(image2, "t={}".format(time2),
    #                (0, 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
    #     cv.putText(image2, "Conf={:.1f}%".format(cells2.at[0, "confidence-2"] * 100),
    #                (0, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
    #     path1 = generate_file_name(template, "", self.FOV, trench, frame1, mode=template_mode)
    #     image1 = cv.imread(image_dir + os.path.sep + path1)
    #     if isinstance(fluores, int) and fluores != 0:
    #         image1 *= fluores
    #     self.image_width = image1.shape[1]
    #     cv.putText(image1, "t={}".format(time1),
    #                (0, 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
    #     landscape = np.concatenate((image1, image2), axis=1)
    #     for c in range(len(cells2.at[0, "label"])):
    #         if cells2.at[0, "parent_label-2"][c] is not None:
    #             parent = int(cells2.at[0, "parent_label-2"][c]) - 1
    #             position1 = (round(cells2.at[0, "centroid"][c][0] + offset + image1.shape[1]),
    #                          round(cells2.at[0, "centroid"][c][1]))
    #             position2 = (round(cells1.at[0, "centroid"][parent][0] + offset),
    #                          round(cells1.at[0, "centroid"][parent][1]))
    #             cv.line(landscape, position1, position2, (0, 255, 0), thickness=2)
    #     plt.figure(figsize=(5, 20))
    #     plt.imshow(landscape)
    #     if save_dir:
    #         write_path = generate_file_name(template, "second_classification_", self.FOV, trench, "", mode=template_mode)
    #         if not os.path.isdir(save_dir):
    #             os.mkdir(save_dir)
    #         cv.imwrite(save_dir + os.path.sep + write_path, landscape)
    #         print(f"saved as {save_dir + os.path.sep + write_path}")
