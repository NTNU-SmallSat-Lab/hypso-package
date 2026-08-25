import os
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from pathlib import Path

labels = {'water': 0,
        'strange_water': 1,
        'light_forest': 2,
        'dark_forest': 3,
        'urban': 4,
        'rock': 5,
        'ice': 6,
        'sand': 7,
        'thick_clouds': 8,
        'thin_clouds': 9,
        'shadows': 10}

def decode_jonas_svm_labels(file_path: Path, spatial_dimensions: tuple[int, int]) -> np.ndarray:

    # Open the binary file and read its content
    with open(file_path, 'rb') as fileID:
        fileContent = fileID.read()

    # Extract the required values from the binary data
    classification_execution_time = int.from_bytes(fileContent[0:4], byteorder='little', signed=True)
    loading_execution_time = int.from_bytes(fileContent[4:8], byteorder='little', signed=True)

    classes_holder = fileContent[8:24]
    labels_holder = fileContent[24:]
    classes = []
    labels = []

    # Decode the labels and convert them back to original classes.
    for i in range(len(classes_holder)):
        if classes_holder[i] != 255:
            classes.append(classes_holder[i])
    if len(classes) <= 2:
        for i in range(len(labels_holder)):
            pixel_str = format(labels_holder[i], '08b')
            for j in range(8):
                labels.append(int(pixel_str[j]))
    if 2 < len(classes) <= 4:
        for i in range(len(labels_holder)):
            pixel_str = format(labels_holder[i], '08b')
            for j in range(4):
                labels.append(int(pixel_str[2 * j:2 * j + 2], 2))
    if 4 < len(classes) <= 16:
        for i in range(len(labels_holder)):
            pixel_str = format(labels_holder[i], '08b')
            for j in range(2):
                labels.append(int(pixel_str[4 * j:4 * j + 4], 2))

    # Corrected label conversion
    for i in range(len(labels)):
        labels[i] = classes[labels[i]]

    # Save 'labels' as a CSV file with a comma delimiter
    # with open('labels.csv', 'w') as csv_file:
    #    csv_file.write(','.join(map(str, labels)))

    data = np.asarray(labels)
    data = data.reshape(spatial_dimensions)

    return data



def decode_jonas_svm_water_mask(file_path: Path, spatial_dimensions: tuple[int, int]) -> np.ndarray:

    data = decode_jonas_svm_labels(file_path=file_path, spatial_dimensions=spatial_dimensions)

    return ((data == labels['water']) | (data == labels['strange_water']))


def decode_jonas_svm_land_mask(file_path: Path, spatial_dimensions: tuple[int, int]) -> np.ndarray:

    data = decode_jonas_svm_labels(file_path=file_path, spatial_dimensions=spatial_dimensions)

    return (data == labels['light_forest'] | \
                    labels['dark_forest'] | \
                    labels['urban'] | \
                    labels['rock'] | \
                    labels['sand'])

    #return (data == labels['water']) | (data == labels['strange_water'])


def decode_jonas_svm_cloud_mask(file_path: Path, spatial_dimensions: tuple[int, int]) -> np.ndarray:

    data = decode_jonas_svm_labels(file_path=file_path, spatial_dimensions=spatial_dimensions)

    return ((data == labels['thick_clouds']) | (data == labels['thin_clouds']))

