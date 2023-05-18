import cv2 as cv
from geojson import Polygon, FeatureCollection, Feature
import numpy as np


def pretize_text(annotation_type):
    if annotation_type == 'blood_vessels':
        return 'Blood vessels'
    elif annotation_type == 'fatty_tissues':
        return 'Fatty tissue'
    elif annotation_type == 'inflammations':
        return 'Inflammation'
    elif annotation_type == 'endocariums':
        return 'Endocarium'
    elif annotation_type == 'fibrotic_tissues':
        return 'Fibrotic tissue'
    elif annotation_type == 'quilities':
        return 'Quilty'
    elif annotation_type == 'immune_cells':
        return 'Immune cells'
    else:
        annotation_type = annotation_type.replace('_', ' ')
        return annotation_type.replace(annotation_type[0], annotation_type[0].upper(), 1)


def get_color(name):
    if name == 'blood_vessels':
        return [
            128,
            179,
            179
        ]
    elif name == 'endocariums':
        return [
            240,
            154,
            16
        ]
    elif name == 'inflammations':
        return [
            255,
            255,
            153
        ]


def get_coors(contour):
    coors = []
    for idx in range(len(contour)):
        coors.append(contour[idx, 0].tolist())

    return coors


def fix_polygon(contour):
    return np.concatenate((contour, [contour[0]]))


def create_properties_template(annotation):
    return {
        "object_type": "annotation",
        "classification": {
            "name": pretize_text(annotation),
            "color": get_color(annotation),
        },
    }


def get_features(contours, annotation):
    features = []
    for contour in contours:
        contour = fix_polygon(contour)
        coors = get_coors(contour)
        if len(coors) <= 2:
            continue

        features.append(Feature(
            geometry=Polygon([coors]),
            properties=create_properties_template(annotation)
        ))

    return features


def create_geojson(mask, annotation_classes=None):
    if annotation_classes is None:
        annotation_classes = [
            'blood_vessels',
            'endocariums',
            'fatty_tissues',
            'fibrotic_tissues',
            'immune_cells',
            'inflammations',
            'quilties'
        ]

    mask = np.uint8(mask * 255)

    features = []
    if len(mask.shape) == 3:
        _, _, classes = mask.shape
        assert classes == len(annotation_classes)

        for c in range(classes):
            contours, hierarchy = cv.findContours(
                mask[:, :, c], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            features.extend(get_features(contours, annotation_classes[c]))

        return FeatureCollection(features)
    else:
        assert len(annotation_classes) == 1

        contours, hierarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        features = get_features(contours, annotation_classes[0])

        return FeatureCollection(features)
