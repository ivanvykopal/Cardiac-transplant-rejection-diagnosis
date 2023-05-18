import cv2 as cv
import numpy as np
import rtree
import geopandas as gpd
from .create_geojson import create_geojson


def get_mask_index(feature, classes):
    class_type = feature['properties']['classification']['name']

    for idx, name in enumerate(classes):
        if class_type.lower() == name.lower():
            return idx

    # else return Other cells
    return 0


def get_mask(shape, annotations, classes):
    x, y = int(shape[0]), int(shape[1])

    classes_masks = [
        np.zeros((x, y, 1), dtype='uint8')
        for _ in range(len(classes))
    ]

    for feat in annotations:
        geometry_name = 'geometry'
        coors = feat[geometry_name]['coordinates'][0]
        pts = [[round(c[0]), round(c[1])] for c in coors]
        cv.fillPoly(
            classes_masks[get_mask_index(feat, classes)],
            [np.array(pts)],
            1
        )

    mask = np.concatenate(classes_masks, axis=2)
    return mask


def postprocess_mask(mask, dilate=False):
    kernel_size = (51, 51)
    area_treshold = 2500
    isBinary = False
    if len(mask.shape) == 2:
        isBinary = True
        mask = np.expand_dims(mask, axis=-1)

    for idx in range(mask.shape[-1]):
        img_modified = mask[:, :, idx]

        # remove small objects
        img_modified = cv.morphologyEx(
            img_modified, cv.MORPH_OPEN, kernel_size)

        # remove horizontal lines
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (75, 5))
        detected_lines = cv.morphologyEx(
            img_modified, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv.findContours(
            detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv.drawContours(img_modified, [c], -1, (0, 0, 0), 5)

        # remove contours with less than 5 points
        contours, _ = cv.findContours(
            img_modified, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 5:
                cv.fillPoly(
                    img_modified,
                    [cnt],
                    0
                )

        img_modified = cv.GaussianBlur(img_modified, kernel_size, 0)

        contours, _ = cv.findContours(
            img_modified, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < area_treshold:
                cv.fillPoly(
                    img_modified,
                    [cnt],
                    0
                )

        img_modified = cv.morphologyEx(
            img_modified, cv.MORPH_OPEN, kernel_size * 5)
        if dilate:
            img_modified = cv.dilate(img_modified, kernel_size * 5, 0)

        mask[:, :, idx] = img_modified

    if isBinary:
        mask = mask[:, :, 0]
    return mask


def postprocess_endocard(gdf, gdf_tissue):
    for idx, row in gdf.iterrows():
        if row['classification']['name'] == 'Endocarium' and gdf_tissue.boundary.distance(row.geometry.boundary).min() > 100:
            gdf.drop(idx, inplace=True)

    return gdf


def postprocess_inflammation(gdf, gdf_immune):
    for idx, row in gdf.iterrows():
        if row['classification']['name'] == 'Inflammation' and gdf_immune.within(row.geometry).sum() < 10:
            gdf.drop(idx, inplace=True)

    return gdf


def postprocess_vessels(gdf, gdf_tissue):
    for idx, row in gdf.iterrows():
        if row['classification']['name'] == 'Blood vessels' and any(gdf_tissue.boundary.intersects(row.geometry.boundary)):
            gdf.drop(idx, inplace=True)

    return gdf


def find_distance(gdf_endocard):
    idx = rtree.index.Index()
    for i, row in gdf_endocard.iterrows():
        idx.insert(i, row.geometry.bounds)

    for i, row in gdf_endocard.iterrows():
        for j in idx.intersection(row.geometry.buffer(15).bounds):
            if i < j and row.geometry.distance(gdf_endocard.loc[j, 'geometry']) < 15:
                return True

    return False


def merge_nearest_endocards(mask):
    isBinary = False
    if len(mask.shape) == 2:
        endocard_mask = mask
        isBinary = True
    else:
        endocard_mask = mask[:, :, -1]

    struct_elem = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    dilate_count = 0

    while True:
        geojson_file = create_geojson(endocard_mask, [
            "endocariums"
        ])
        gdf_endocard = gpd.GeoDataFrame.from_features(geojson_file)

        distance = find_distance(gdf_endocard)
        if distance:
            endocard_mask = cv.dilate(endocard_mask, struct_elem)
            dilate_count += 1
        else:
            break

    for _ in range(dilate_count):
        endocard_mask = cv.erode(endocard_mask, struct_elem)

    if isBinary:
        mask = endocard_mask
    else:
        mask[:, :, -1] = endocard_mask

    return mask
