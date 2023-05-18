import argparse
import glob
import numpy as np
import json
import os

from Utils.Postprocessing.create_geojson import create_geojson


def concat_inference(output_directory, threshold=0.2):
    file_idx = 0
    with open('./data/images_256.json') as json_file:
        train_valid_json = json.load(json_file)

    files = glob.glob(f'{output_directory}/*_HE')

    for idx, f in enumerate(files[file_idx:]):
        file_name = f.replace('\\', '/').split('/')[-1]
        subfiles = glob.glob(f'{f}/*')

        image = train_valid_json['images'][idx + file_idx]
        canvas = np.zeros((image['height'], image['width'], 3), dtype=np.float32)

        f_name = f.replace('\\', '/')
        print(f_name)
        if not os.path.isdir(f_name):
            continue

        for subfile in subfiles:
            name = subfile.replace('\\', '/').replace(f"{f_name}/", '').replace('.npy', '')
            fragment_idx, y, x = name.split('_')
            y, x = int(y), int(x)
            try:
                mask = np.load(subfile)
                canvas[
                    y:y + mask.shape[0],
                    x:x + mask.shape[1]
                    ] = mask[:512 + (canvas.shape[0] - y - mask.shape[0]), :512 + (canvas.shape[1] - x - mask.shape[1]), :]
            except Exception:
                print(y, x)

        canvas = np.array(canvas > threshold, dtype=np.uint8)
        # Všetky ostatné modely majú threshold nastavený na 0.2
        geojson_file = create_geojson(canvas, [
            "blood_vessels",
            "inflammations",
            "endocariums",
        ])
        with open(f"{output_directory}/{file_name}.geojson", "w") as outfile:
            json.dump(geojson_file, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.2, help='Threshold')
    parser.add_argument('--output_directory', type=str, help='Directory with saved predictions')

    args = parser.parse_args()
    concat_inference(
        output_directory=args.output_directory,
        threshold=args.threshold
    )
