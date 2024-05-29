import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

from datetime import datetime
from pathlib import Path
from typing import Tuple

RECON_FACE_PATH = Path(__file__).parent / 'reconstructed_face/'

def compute_svd(X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U, S, Vt


def compute_alpha(x: np.ndarray, U: np.ndarray, r: int) -> np.ndarray: 
    alpha = U[:, :r].T @ x
    return alpha


def reconstruct_test_face(alpha: np.ndarray, U: np.ndarray, avg_face: np.ndarray, r: int) -> None: 
    recon_face = U[:, :r] @ alpha
    recon_face = recon_face.reshape(recon_face.shape[0], 1) + avg_face
    img = plt.imshow(np.reshape(recon_face, (192, 168)))
    img.set_cmap('gray')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = os.path.join(RECON_FACE_PATH, f'recon_face_{timestamp}.png')
    plt.savefig(output_path)