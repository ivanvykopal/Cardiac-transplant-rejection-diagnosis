import json
from sklearn.utils import shuffle


def train_test_split(data_name, config, split1=None, split2=None):
    with open(data_name) as json_file:
        train_valid_json = json.load(json_file)

    files = shuffle(train_valid_json['images'], random_state=42)
    if split1 is None:
        split1 = int(0.7 * len(files))

    if split2 is None:
        split2 = int(0.5 * (len(files) - split1))

    train_filenames = files[:split1]
    valid_filenames = files[split1:split1 + split2]
    test_filenames = files[split1 + split2:]

    train_json = {
        "images": train_filenames,
        "patch_size": train_valid_json['patch_size']
    }

    valid_json = {
        "images": valid_filenames,
        "patch_size": train_valid_json['patch_size']
    }

    test_json = {
        "images": test_filenames,
        "patch_size": train_valid_json['patch_size']
    }

    size = 0
    for image in train_json['images']:
        size += len(image['patches'])

    train_size = size * (len(config['augmentations']) + 1)

    return {"data": train_json, "size": train_size}, {"data": valid_json}, {"data": test_json}


def train_test_split_fragments(data_name, config, split1=None, split2=None):
    with open(data_name) as json_file:
        train_valid_json = json.load(json_file)

    files = []
    for image in train_valid_json['images']:
        for idx, fragment in enumerate(image['fragments']):
            for patch in fragment['patches']:
                files.append({
                    "patch": patch,
                    "name": image['name'],
                    "height": image['height'],
                    "width": image['width'],
                    "fragment": {
                        "idx": idx,
                        "x": fragment['x'],
                        "y": fragment['y'],
                        "width": fragment['width'],
                        "height": fragment['height']
                    }
                })

    files = shuffle(files, random_state=42)
    if split1 is None:
        split1 = int(0.7 * len(files))

    if split2 is None:
        split2 = int(0.5 * (len(files) - split1))

    train_filenames = files[:split1]
    valid_filenames = files[split1:split1 + split2]
    test_filenames = files[split1 + split2:]

    size = len(train_filenames)

    train_size = size * (len(config['augmentations']) + 1)

    return {"data": train_filenames, "size": train_size}, {"data": valid_filenames}, {"data": test_filenames}
