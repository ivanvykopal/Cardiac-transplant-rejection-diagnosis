import bioformats as bf
import copy
import cv2 as cv
import geojson
from geojson import Polygon, FeatureCollection, Feature, dump
import geopandas as gpd
from glob import glob
import javabridge
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import shape, Polygon
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from args import parser


def set_logs():
    myloglevel = "ERROR"
    rootLoggerName = javabridge.get_static_field(
        "org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", myloglevel,
                                           "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel",
                    "(Lch/qos/logback/classic/Level;)V", logLevel)


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


def find_tissues(gj):
    features = gj['features']

    for feature in features:
        class_type = feature['properties']['classification']['name']
        if class_type == 'Region*':
            return feature


def create_polygon(coors):
    return {
        "type": "Polygon",
        "coordinates": coors
    }


def get_fragment_coords(gj):
    if gj:
        tissues = find_tissues(gj)['geometry']['coordinates']
        main_features_len = len(tissues)
        main_coordinates = list()

        for i in range(0, main_features_len):
            coordinates = tissues[i]
            geo: dict = {'type': 'Polygon', 'coordinates': coordinates}
            polygon: Polygon = shape(geo)
            mabr = polygon.bounds
            mabr = [int(x) for x in mabr]
            main_coordinates.append(mabr)

        return main_coordinates, gj
    else:
        return None, None


def get_geojson_from_file(path):
    try:
        return geojson.load(open(path))
    except OSError:
        return None


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
        if len(coors) < 3:
            continue

        features.append(Feature(
            geometry=create_polygon([coors]),
            properties=create_properties_template(annotation)
        ))

    return features


def create_geojson(mask, annotation_classes):
    mask = np.uint8(mask)

    features = []
    if len(mask.shape) == 3:
        _, _, classes = mask.shape
        assert classes == len(annotation_classes)

        for c in range(classes):
            contours, _ = cv.findContours(
                mask[:, :, c], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            features.extend(get_features(contours, annotation_classes[c]))

        return FeatureCollection(features)
    else:
        assert len(annotation_classes) == 1

        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        features = get_features(contours, annotation_classes[0])

        return FeatureCollection(features)


# later we should change nucleusGeometry to geometry
def get_annotations(coordinates, gj):
    features = gj['features']

    polygons = dict()
    for a in range(len(coordinates)):
        polygons[a] = []

    for poly in features[1:]:
        akt_poly = copy.deepcopy(poly)
        if len(poly['nucleusGeometry']['coordinates'][0][0]) == 2:
            test_coords_x, test_coords_y = poly['nucleusGeometry']['coordinates'][0][0]
            for idx, (left, bottom, right, top) in enumerate(coordinates):
                if right >= test_coords_x >= left and bottom <= test_coords_y <= top:
                    coordinates_poly = akt_poly['nucleusGeometry']['coordinates']
                    shifted_coords = [(int(coords[0]) - left, int(coords[1]) - bottom) for coords in
                                      coordinates_poly[0]]
                    if shifted_coords[0][0] > 0 and shifted_coords[0][1] > 0:
                        akt_poly['nucleusGeometry']['coordinates'] = [
                            shifted_coords]
                        polygons[idx].append(akt_poly)

    return polygons


def get_area(contours):
    area = 0
    for contour in contours:
        area += cv.contourArea(contour)

    return area


def dilate(mask):
    dilated = mask
    nuclei, _ = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    num_dilatations = 0

    while len(nuclei) != 1:
        dilated = cv.dilate(dilated, cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (10, 10)))
        nuclei, _ = cv.findContours(
            dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        num_dilatations += 1

    return dilated


def ckd_nearest(gd_a):
    n_a = np.array(list(gd_a.geometry.centroid.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(n_a)
    dist, idx = btree.query(n_a, k=2)

    return dist[:, 1]


def iterative_dilation(image1, image2, distance_threshold, only_immune=False):
    nuclei, _ = cv.findContours(
        image1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    nuclei_count_all = len(nuclei)
    nuclei_count = nuclei_count_all
    nuclei_count_diff = nuclei_count_all

    dilated1 = image1
    dilated2 = image2
    while nuclei_count_diff:
        dilated1 = cv.dilate(
            dilated1, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        if not only_immune:
            dilated2 = cv.dilate(
                dilated2, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

        contours, _ = cv.findContours(
            dilated1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        gdf = gpd.GeoDataFrame.from_features(
            FeatureCollection(get_features(contours, 'other')))

        distances = ckd_nearest(gdf)
        if all([distance > distance_threshold for distance in distances]):
            break

        nuclei_count_diff = nuclei_count - len(contours)
        nuclei_count = len(contours)

    return dilated1, dilated2


def get_inflammatory_clustering(image_shape, gj, eps=100, min_samples=15, only_immune=False):
    features = gj['features']
    centroids = list()
    polygons = dict()

    index = 0
    for feature in features:
        if feature['properties']['classification']['name'] != 'Region*':
            if only_immune and feature['properties']['classification']['name'] == 'Immune cells':
                s = shape(feature['geometry'])
                polygons[index] = s
                centroids.append([s.centroid.x, s.centroid.y])
                index += 1
            elif not only_immune:
                s = shape(feature['geometry'])
                polygons[index] = s
                centroids.append([s.centroid.x, s.centroid.y])
                index += 1

    X = np.array(centroids)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    unique = np.unique(db.labels_)
    dilated_img = np.zeros(image_shape).astype(np.uint8)

    for unique_idx, unique_value in enumerate(unique[1:]):
        indexes = np.where(db.labels_ == unique_value)[0]
        mask = np.zeros(image_shape, dtype=np.uint8)

        for idx in indexes:
            coors = list(zip(*polygons[idx].exterior.coords.xy))
            pts = [[round(c[0]), round(c[1])] for c in coors]
            cv.fillPoly(mask, [np.array(pts)], 1)

        mask = dilate(mask)
        dilated_img[mask != 0] = 1

    return dilated_img


def get_inflammatory_dilation(image_shape, gj, distance_threshold=50, only_immune=False):
    image_immune = np.zeros(image_shape, dtype=np.uint8)
    image_other = np.zeros(image_shape, dtype=np.uint8)

    for feature in gj['features']:
        if feature['properties']['classification']['name'] == 'Immune cells':
            polygon = feature['geometry']
            for coors in polygon['coordinates']:
                pts = [[round(c[0]), round(c[1])] for c in coors]
                cv.fillPoly(image_immune, [np.array(pts)], 1)
        elif not only_immune and feature['properties']['classification']['name'] != 'Region*':
            polygon = feature['geometry']
            for coors in polygon['coordinates']:
                pts = [[round(c[0]), round(c[1])] for c in coors]
                cv.fillPoly(image_other, [np.array(pts)], 1)

    dilated_immune, dilated_other = iterative_dilation(
        image_immune,
        image_other,
        distance_threshold,
        only_immune=only_immune
    )

    return dilated_immune, dilated_other


def create_mask_from_df(df, size):
    canvas = np.zeros(size).astype(np.uint8)
    for index, row in df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            polygon = row['geometry']
            coors = list(zip(*polygon.exterior.coords.xy))
            pts = [[round(c[0]), round(c[1])] for c in coors]
            cv.fillPoly(canvas, [np.array(pts)], 1)

    return canvas


def get_final_geojson(df1, df2):
    for index, row in df2.iterrows():
        try:
            row_intersections = df1[df1.intersects(
                row['geometry'].buffer(0))].buffer(0)
        except:
            continue

        if row_intersections is not None and len(list(row_intersections.index)) > 0:
            df1.loc[row_intersections.index, 'geometry'] = \
                df1.loc[row_intersections.index, 'geometry'].union(
                    row['geometry'].buffer(0))

    return df1


def process_image(args):
    javabridge.start_vm(class_path=bf.JARS)
    set_logs()
    if os.path.isdir(args.image_path) and os.path.isdir(args.geojson_path):
        files = glob(f'{args.image_path}/*.vsi')
        geojsons = glob(f'{args.geojson_path}/*.geojson')
    else:
        files = [args.image_path]
        geojsons = [args.geojson_path]

    for file in files:
        file_name = file.replace('\\', '/').split('/')[-1].split('.')[0]
        print(f'Analyzing {file_name}!')

        geojson_file = [geo for geo in geojsons if file_name in geo]
        if len(geojson_file) == 0:
            print(f'{file_name} does not have appropriate geojson file!')
            continue
        else:
            geojson_file = geojson_file[0]

        image_reader = bf.formatreader.make_image_reader_class()()
        gj = get_geojson_from_file(geojson_file)

        if gj is None:
            print('Error during reading geojson file!')
            return None

        image_reader.allowOpenToCheckType(True)
        image_reader.setId(file)
        image_reader.setSeries(0)
        wrapper = bf.formatreader.ImageReader(path=file, perform_init=False)
        wrapper.rdr = image_reader

        sizeX = wrapper.rdr.getSizeX()
        sizeY = wrapper.rdr.getSizeY()

        if args.algorithm == 'dbscan':
            mask = get_inflammatory_clustering(
                (sizeY, sizeX), gj, min_samples=args.num_cells, eps=args.cell_distance, only_immune=args.only_immune
            )
            geojson_file = create_geojson(mask, ['inflammations'])
            gdf = gpd.GeoDataFrame.from_features(geojson_file)
            if not gdf.empty:
                gdf = gdf[gdf.area > args.min_area]
                gdf.to_file(
                    f'{args.output_path}/{file_name}.geojson', driver='GeoJSON')

        elif args.algorithm == 'dilatation':
            mask1, mask2 = get_inflammatory_dilation(
                (sizeY, sizeX),
                gj,
                distance_threshold=args.cell_distance,
                only_immune=args.only_immune
            )
            geojson_file1 = create_geojson(mask1, ['inflammations'])
            df1 = gpd.GeoDataFrame.from_features(geojson_file1)
            if not args.only_immune:
                geojson_file2 = create_geojson(mask2, ['inflammations'])
                df2 = gpd.GeoDataFrame.from_features(geojson_file2)
                if not df1.empty:
                    df = get_final_geojson(df1, df2)
                    df = df[df.area > args.min_area]
                    mask = create_mask_from_df(df, (sizeY, sizeX))
                else:
                    mask = mask1
            else:
                if not df1.empty:
                    df1 = df1[df1.area > args.min_area]
                    mask = create_mask_from_df(df1, (sizeY, sizeX))
                else:
                    mask = mask1

            with open(f'{args.output_path}/{file_name}.geojson', 'w') as f:
                dump(create_geojson(mask, ['inflammations']), f)

        print(f'Analyzing of {file_name} is done!')

    if len(files) > 1:
        print(f'Analyzing is done!')
    javabridge.kill_vm()


if __name__ == '__main__':
    args = parser.parse_args()
    process_image(args)

# dbscan
# distance = 100
# area threshold = 50_000

# dilatation
# distance = 45
# area threshold = 40_000
