import numpy as np
import pandas as pd
import os

from PIL import Image
from typing import List
from typing import Optional
from dataclasses import dataclass 


@dataclass
class XInput: 
    X: np.ndarray
    X_test: List[np.ndarray]
    avg_face: np.ndarray


def load_and_flatten_image(path: str) -> np.ndarray: 
    with Image.open(path) as img:
        img_array = np.array(img)
        return img_array.flatten()


def data_loader() -> List[np.ndarray]: 
    all_faces = []
    for subdir, _, files in os.walk('all_faces/'):
        for file in files: 
            if file.endswith('.pgm'):
                full_path = os.path.join(subdir, file)
                all_faces.append(load_and_flatten_image(full_path))
    return all_faces


def get_avg_face(X: np.ndarray) -> np.ndarray:
    avg_face = np.mean(X, axis=1)
    return avg_face[:, np.newaxis]


def construct_X(test_holdout: Optional[int] = 0) -> XInput: 
    all_faces = data_loader()
    X = np.column_stack(all_faces)

    # Get average face and center X matrix
    avg_face = get_avg_face(X)
    X_centered = X - avg_face

    # Extract test columns
    if test_holdout > 0:
        test = X_centered[:, :test_holdout]
        test_list = [test[:, i] for i in range(test.shape[1])]

        return XInput(
            X_centered[:, :-test_holdout], 
            test_list,
            avg_face
        )
    else: 
        return XInput(
            X_centered, 
            None,
            avg_face
        )
