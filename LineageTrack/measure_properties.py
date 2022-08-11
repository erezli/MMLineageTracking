import numpy as np
import tifffile
from skimage.measure import regionprops_table, label
from skimage.io import imshow, imread
from skimage.morphology import remove_small_objects
import pandas as pd


### --------------------------------------------from Charlie------------------------------------------------------- ###
def get_cell_props(label_img_path, intensity_img_path, min_size=80):
    """
    Returns the region properties specified in this function for label and intensity image pair.
    Objects smaller than a minimum size are unlikely to be cells and are removed from the analysis.

    :param label_img_path: The full path to the label image file (a binary mask).
    :type label_img_path: class: 'str'
    :param intensity_img_path: The full path to the intensity image file you wish measure the properties of.
    :type intensity_img_path: class: 'str'
    :param min_size: The minimum size in pixels of a region to be included in the analysis.
    :type min_size: class: 'int'
    :return: A dictionary of all the cell properties measured in a given image.
    :rtype: class: 'dict'
    """
    # read label image
    label_img = imread(label_img_path)
    label_img = label_img.astype(bool)
    label_img = remove_small_objects(label_img, min_size=min_size)
    labels = label(label_img, connectivity=1)

    # read intensity image
    intensity_img = tifffile.imread(intensity_img_path)

    data = regionprops_table(labels, intensity_img, properties=(
        "area", "major_axis_length", "minor_axis_length", "centroid",
        "intensity_mean", "label", "centroid_local", "image_intensity", "orientation"))
    return data


### --------------------------------------------------------------------------------------------------------------- ###


def add_information(data, channel, trench_id, time, identity):
    data["channel"] = channel
    data["trench_id"] = trench_id
    data["time_(mins)"] = time
    data["identity"] = identity
    return data


def combine_data(data_list):
    all_data = dict()
    for key in data_list[0]:
        all_data[key] = []
        for data in data_list:
            all_data[key].append(data[key])
    df = pd.DataFrame(data)
    return df
