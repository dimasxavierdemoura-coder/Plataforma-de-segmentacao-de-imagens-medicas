import os
import sys
import numpy as np
from PIL import Image
import nibabel as nib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import load_image_file, load_nifti_image


def test_load_image_file_png(tmp_path):
    image_path = tmp_path / "test.png"
    image = Image.new("RGB", (10, 10), color=(123, 234, 45))
    image.save(image_path)

    loaded = load_image_file(str(image_path))
    assert loaded.shape == (10, 10, 3)
    assert loaded.dtype == np.uint8


def test_load_nifti_image_center_slice(tmp_path):
    data = np.zeros((16, 16, 8), dtype=np.float32)
    data[:8, :8, 4] = 1.0
    nifti_file = tmp_path / "test.nii"
    nifti = nib.Nifti1Image(data, np.eye(4))
    nib.save(nifti, str(nifti_file))

    loaded = load_nifti_image(str(nifti_file))
    assert loaded.shape == (16, 16)
    assert loaded.max() == 255
    assert loaded.min() == 0
