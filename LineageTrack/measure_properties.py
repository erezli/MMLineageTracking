import os
import numpy as np
import tifffile
from skimage.measure import regionprops_table, label
from skimage.io import imshow, imread
from skimage.morphology import remove_small_objects
import pandas as pd
from tqdm import tqdm


def get_cell_props(label_img_path, intensity_img_path, min_size=80):    # edited from Charlie's function
    """
    Returns the region properties specified in this function for label and intensity image pair.
    Objects smaller than a minimum size are unlikely to be cells and are removed from the analysis.

    @param label_img_path: The full path to the label image file (a binary mask).
    @type label_img_path: class: 'str'
    @param intensity_img_path: The full path to the intensity image file you wish measure the properties of.
    @type intensity_img_path: class: 'str'
    @param min_size: The minimum size in pixels of a region to be included in the analysis.
    @type min_size: class: 'int'
    @return: A dictionary of all the cell properties measured in a given image.
    @rtype: class: 'dict'
    """
    # read label image
    label_img = tifffile.imread(label_img_path)
    # label_img = label_img.astype(bool)
    label_img = remove_small_objects(label_img, min_size=min_size)
    labels = label(label_img, connectivity=1)
    # labels = label_img

    # read intensity image
    intensity_img = imread(intensity_img_path)

    data = regionprops_table(labels, intensity_img, properties=(
        "area", "major_axis_length", "minor_axis_length", "centroid",
        "intensity_mean", "label", "centroid_local", "image_intensity", "orientation"))
    # make sure it is in the correct order
    order = sorted(range(len(data["centroid-0"])), key=lambda k: data["centroid-0"][k])
    for keys in data:
        data[keys] = [data[keys][i] for i in order]
    return data


def add_information(data, channel, trench_id, time, identity):
    length = len(data["area"])
    data["label"] = [int(i + 1) for i in range(length)]
    data["channel"] = [channel] * length
    data["trench_id"] = [trench_id] * length
    data["time_(mins)"] = [time] * length
    data["identity"] = [identity] * length
    return data


def combine_data(data_list):
    all_data = dict()
    for key in data_list[0]:
        all_data[key] = []
        for data in data_list:
            all_data[key].extend(data[key])
    df = pd.DataFrame(all_data)
    return df


def generate_csv(fov, label_dir, intensity_dir, dt=1, save_dir="./temp/", min_size=50, channels=None, step=1):
    label_images = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
    # if channels is None:
    #     channels = set()
    #     intensity_images = sorted(
    #         [f for f in os.listdir(intensity_dir) if os.path.isfile(os.path.join(intensity_dir, f))])
    #     for img in intensity_images:
    #         channels.add(img.split("_")[1])
    #     print(channels)
    save_loc = []
    # example file name xy000_PC_TR0_T0000_suffix.tif
    for channel in channels:
        data_list = []
        for i in tqdm(range(len(label_images)), desc=f"reading through images in channel {channel} and FOV {fov}"):
            info = label_images[i].split(".")[0].split("_")
            # if int(info[4].split("-")[1]) == 20 and int(info[5].split("-")[1].split(".")[0]) == 99 and info[0] == fov:
            if info[0] == fov:
                # intensity_image = label_images[i].split("-")[0].replace(info[1], channel) + ".tif"
                intensity_image = label_images[i].split(".")[0].replace(info[1], channel) + ".png"
                # suffix = "-".join(label_images[i].split("-")[1:]).split(".")[0]
                trench = int(info[2][2:])
                time = int(info[3][1:])
                if time % step == 0:
                    data = get_cell_props(os.path.join(label_dir, label_images[i]),
                                          os.path.join(intensity_dir, intensity_image), min_size=min_size)

                    data = add_information(data, channel=channel, trench_id=trench, time=time*dt, identity=intensity_image)
                    data_list.append(data)
        df = combine_data(data_list)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        # df.to_csv(os.path.join(save_dir, f"{fov}_{channel}{suffix}.csv"))
        df.to_csv(os.path.join(save_dir, f"{fov}_{channel}.csv"))
        # save_loc.append(os.path.join(save_dir, f"{fov}_{channel}{suffix}.csv"))
        save_loc.append(os.path.join(save_dir, f"{fov}_{channel}.csv"))
    return f"saved to {save_loc}"
