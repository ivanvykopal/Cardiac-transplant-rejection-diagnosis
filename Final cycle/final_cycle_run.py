import argparse
import os
import shutil
import timeit


def run(model_config_paths, image_path, output_path, cell_path, temp_file_path):
    if os.path.exists(temp_file_path):
        shutil.rmtree(temp_file_path)
    os.mkdir(temp_file_path)
    model_config_path = ''
    for path in model_config_paths:
        model_config_path += f'{path} '
    while True:
        result = os.system(
            f'python ./final_cycle.py --model_config_path {model_config_path} --image_path "{image_path}" --output_path "{output_path}" --cell_path "{cell_path}" --metadata_path "{temp_file_path}"')
        if result == 0:
            shutil.rmtree(temp_file_path)
            return()


if __name__ == '__main__':
    model_config_path = ['deeplabv3plus1_final.yaml',
                         'deeplabv3plus2_final.yaml', 'deeplabv3plus3_final.yaml']
    image_name = '1225_21_HE'
    image_path = f'D:/Master Thesis/Data/EMB-IKEM-2022-03-09/{image_name}.vsi'
    cell_path = f'D:/Master Thesis/Data/Cell Annotations/{image_name}.vsi - 20x.geojson'
    output_path = 'D:/Test'
    temp_file_path = './temp'

    parser = argparse.ArgumentParser(
        description='Predict higher morphological structures')
    parser.add_argument('--model_config_path', nargs='+', default=model_config_path,
                        help='Path to the model configs')
    parser.add_argument('--image_path', type=str,
                        help='Path to the image', default=image_path)
    parser.add_argument('--cell_path', type=str,
                        help='Path to the cell mask', default=cell_path)
    parser.add_argument('--output_path', type=str,
                        help='Path to the output folder', default=output_path)
    parser.add_argument('--metadata_path', type=str,
                        help='Path to the metadata folder', default=f'{output_path}/temp')
    parser.add_argument('--overlap', type=bool, default=True)

    run(model_config_path, image_path,
        output_path, cell_path, temp_file_path)
