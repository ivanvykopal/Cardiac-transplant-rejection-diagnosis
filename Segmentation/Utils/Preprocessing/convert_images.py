from tensorflow.keras.utils import load_img
import numpy as np
import glob
from tqdm import tqdm


def img2npy(directory, filename, output_dir):
    img = np.array(load_img(f"{directory}{filename}", color_mode='rgb'))
    filename = filename.replace('.tiff', '.npy')
    np.save(f"{output_dir}{filename}", img)


if __name__ == "__main__":
    directory = 'D:\\Master Thesis\\Code\\Segmentation\\data3\\'

    files = glob.glob(f"{directory}*.tiff")

    for file in tqdm(files, total=len(files)):
        filename = file.replace(directory, '')
        img2npy(directory, filename, directory)
