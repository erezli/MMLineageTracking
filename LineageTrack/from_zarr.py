import numpy as np
from skimage.measure import regionprops_table, label
# from skimage.io import imshow, imread
from skimage.morphology import remove_small_objects, dilation, erosion
import pandas as pd
from tqdm import tqdm
import zarr
import os
from ast import literal_eval
import mahotas
# import cv2 as cv


def get_cell_props(mask, channel_img, min_size=80):
    """
    Returns the region properties specified in this function for label and intensity image pair.
    Objects smaller than a minimum size are unlikely to be cells and are removed from the analysis.

    @param mask: Image array for the mask
    @param channel_img: Image array for the specific channel
    @param min_size: The minimum size in pixels of a region to be included in the analysis.
    @return: A dictionary of all the cell properties measured in a given image.
    """

    ###
    # # not necessary if your masks have good labels  
    # mask = mask.astype(bool)

    # filtered = remove_small_objects(mask, min_size=min_size)

    # width = int(mask.shape[1]/15)
    # height = int(mask.shape[1]/20)
    # X, Y = np.ogrid[0:width*2+1, 0:height*2+1]

    # footprint1 = (1./3 * (X - width)**2 + (Y - height)**2 < height**2).astype(np.uint8)
    # sep = erosion(filtered, footprint1)
    # labels = dilation(label(sep, connectivity=1), footprint1)
    # labels = label(filtered, connectivity=1)
    ###

    labels = mask.astype(int)
    labels = remove_small_objects(labels, min_size=min_size)

    data = regionprops_table(labels, channel_img, properties=(
        "area", "major_axis_length", "minor_axis_length", "centroid",
        "intensity_mean", "intensity_max", "centroid_local", "image_intensity", "orientation"
    ))
    order = sorted(range(len(data["centroid-0"])), key=lambda k: data["centroid-0"][k])
    for keys in data:
        data[keys] = [data[keys][i] for i in order]
    return data


def extract_from_string(array): # for use when reading image intensity array from csv file (it will be string)
    return np.asarray(literal_eval(array.replace( '[   ' , '[' ).replace( '[  ' , '[' )
                                   .replace( '[ ' , '[' ).replace( '    ' , ',' )
                                   .replace( '   ' , ',' ).replace( '  ' , ',' )
                                   .replace( ' ' , ',' )), dtype=np.float32)


def add_information(data, channel, trench_id, time, identity, descriptor, radius):
    length = len(data["area"])
    data["label"] = [int(i + 1) for i in range(length)]
    data["channel"] = [channel] * length
    data["trench_id"] = [trench_id] * length
    data["time_(mins)"] = [time] * length
    data["identity"] = [identity] * length
    if channel == 'mVenus': #temporary adjustment
        image_list = [im for im in data["image_intensity"]]
        data["intensity_total"] = [np.sum(im) for im in image_list]
    if descriptor and channel == 'PC':
        # image_list = [extract_from_string(im) for im in data["image_intensity"]]
        image_list = [im for im in data["image_intensity"]]
        # data["haralick"] = [mahotas.features.haralick(im) for im in image_list]
        # data["haralick_half"] = [(mahotas.features.haralick(im[:int(im.shape[0]/2), :]),
        #                           mahotas.features.haralick(im[int(im.shape[0]/2):, :])) for im in image_list]
        # make sure radius covers the whole array
        data["zernike"] = [list(mahotas.features.zernike_moments(im.astype(bool), radius)) for im in image_list]
        data["zernike_half"] = [[list(mahotas.features.zernike_moments(im.astype(bool)[:int(im.shape[0]/2), :], radius)),
                                 list(mahotas.features.zernike_moments(im.astype(bool)[int(im.shape[0]/2):, :], radius))]
                                for im in image_list]
    return data


def combine_data(data_list):
    all_data = dict()
    for key in data_list[0]:
        all_data[key] = []
        for data in data_list:
            all_data[key].extend(data[key])
    df = pd.DataFrame(all_data)
    return df


def generate_csv(mask_path, img_path, save_dir, dt=1, min_size=0, 
                 channels=[0], c_names=[None], step=1, descriptor=False, radius=None):
    """
    Generate cell properties in CSV files for each channel, using the zarr array of channel images and masks.

    @param mask_path: Path to mask zarr array, expected shape is (trench, time, height, width)
    @param img_path: Path to image zarr array, expected shape is (trench, time, channel, height, width)
    @param save_dir: Path to the save directory
    @param dt: The time interval in minutes
    @param min_size: The minimum size in pixels of a region to be included in the analysis
    @param channels: List of indices to the channel you want to extract
    @param c_names: List of the corresponding channel names
    @param step: The step to take when looping through images over time (for downsampling purposes)
    @param descriptor: Specify whether to extract descriptor information or not (Zernike and Haralick)
    @param radius: radius input for Zernike polynomial
    """
    save_loc = []
    z1 = zarr.open(mask_path, mode='r')
    z2 = zarr.open(img_path, mode='r')
    for c, n in zip(channels, c_names):
        data_list = []
        for i in tqdm(range(z1.shape[0]), 
                      desc=f"reading through images in channel {n}..."):
            for j in range(z1.shape[1]):
                mask_image = z1[i, j, 0, :, :] # may need to change
                intensity_image = z2[i, j, c, :, :]
                trench = i
                time = j
                if time % step == 0:
                    data = get_cell_props(mask=mask_image, 
                                          channel_img=intensity_image, 
                                          min_size=min_size)
                    data = add_information(data, 
                                           channel=n,
                                           trench_id=trench,
                                           time=time*dt,
                                           identity=None,
                                           descriptor=descriptor,
                                           radius=radius)   # from json file
                    data_list.append(data)
        df = combine_data(data_list)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        df.to_csv(os.path.join(save_dir, f"{n}_properties.csv"))
        save_loc.append(os.path.join(save_dir, f"{n}_properties.csv"))
    return f"saved to {save_loc}"