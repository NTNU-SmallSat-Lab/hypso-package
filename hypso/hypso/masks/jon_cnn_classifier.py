import os
from pathlib import Path
import numpy as np

labels = {'water': 0,
          'land': 1,
          'cloud': 2}


def decode_jon_cnn_labels(file_path: Path, spatial_dimensions: tuple[int, int]):

    #file_name = os.path.join(path, 'jon-cnn.labels')

    with open(file_path, mode='rb') as file: # b is important -> binary
        file_content = file.read()

    data = np.frombuffer(file_content, dtype=np.uint8)

    data = data.reshape(spatial_dimensions)

    return data

def decode_jon_cnn_water_mask(file_path: Path, spatial_dimensions: tuple[int, int]) -> np.ndarray:

    data = decode_jon_cnn_labels(file_path=file_path, spatial_dimensions=spatial_dimensions)

    return (data == labels['water'])


def decode_jon_cnn_land_mask(file_path: Path, spatial_dimensions: tuple[int, int]) -> np.ndarray:

    data = decode_jon_cnn_labels(file_path=file_path, spatial_dimensions=spatial_dimensions)

    return (data == labels['land'])


def decode_jon_cnn_cloud_mask(file_path: Path, spatial_dimensions: tuple[int, int]) -> np.ndarray:

    data = decode_jon_cnn_labels(file_path=file_path, spatial_dimensions=spatial_dimensions)

    return (data == labels['cloud'])