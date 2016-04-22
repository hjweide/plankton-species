import numpy as np
from PIL import Image


def load_image(filename, resize_to=None):
    img = Image.open(filename)
    if resize_to is not None:
        h, w = resize_to
        img = img.resize((w, h), Image.ANTIALIAS)
    img = np.asarray(img, dtype=np.float32)

    return img


def transform(X, y):
    X_copy = X.copy()
    y_copy = y.copy()

    v_idx = np.random.choice([True, False], replace=True, size=X.shape[0])
    h_idx = np.random.choice([True, False], replace=True, size=X.shape[0])

    X_copy[v_idx] = X_copy[v_idx, :, ::-1, :]
    X_copy[h_idx] = X_copy[h_idx, :, :, ::-1]

    return X_copy, y_copy
