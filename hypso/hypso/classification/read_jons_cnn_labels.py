import os
from pathlib import Path
import numpy as np

labels = {'water': 0,
          'land': 1,
          'cloud': 2}


def decode_svm_labels(path: Path, spatial_dimensions: tuple):

    file_name = os.path.join(path, 'jon-cnn.labels')

    with open(file_name, mode='rb') as file: # b is important -> binary
        file_content = file.read()

    data = np.frombuffer(file_content, dtype=np.uint8)

    data = data.reshape(spatial_dimensions)

    return data

def decode_svm_labels_masks(path: Path, spatial_dimensions: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    data = decode_svm_labels(path=path, spatial_dimensions=spatial_dimensions)

    water_mask = data == labels['water']
    land_mask = data == labels['land']
    cloud_mask = data == labels['cloud']

    return water_mask, land_mask, cloud_mask

