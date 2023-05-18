from Utils.Preprocessing.MaskCreator import MaskCreator
from Utils.Preprocessing.HasuraClient import HasuraClient

from tqdm import tqdm
import json
import glob


def preprocess_data(json_file, directory, url, header):
    mask_creator = MaskCreator(url, header)

    for image in tqdm(json_file['images']):
        name = image['name']
        npy_name = name + '.npy'
        mask_creator.create_mask_from_DB(image, directory, npy_name)


def import_data_to_DB(url, header, data_directory):
    client = HasuraClient(url, header)
    files = glob.glob(data_directory + '\\*.geojson')

    for file in files:
        file_name = file.replace(data_directory + '\\', '')
        index = file_name.index('.')
        file_name = file_name[:index]

        with open(file) as file:
            json_data = json.loads(file.read())
            client.import_annotations_to_DB(json_data, file_name)
