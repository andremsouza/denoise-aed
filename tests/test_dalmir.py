import numpy as np
from src.dalmir.dalmir import Dalmir


def read_fvecs(file_path):
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
        dim = data[0]
        data = data.reshape(-1, dim + 1)
        return data[:, 1:].astype("float32")


# X = read_fvecs("../siftsmall/siftsmall_learn.fvecs")
# dalmir = Dalmir(normalize=True)
# intrinsic_dim = dalmir(X, method="2nn")
# print(f"Intrinsic Dimension: {intrinsic_dim}")
