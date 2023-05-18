import pandas as pd
import geopandas as gpd
import geojson
import glob
from shapely import wkb
from shapely.geometry import Polygon, mapping


class Geopandas:
    def __init__(
            self,
            annotation_path=r'C:/Users/ivanv/Desktop/Ivan/FIIT/02_ING/Master thesis/Code/Segmentation/data/3-fold cv/Annotations',
            vessels_buffer=2
    ):
        self._load_annotation(annotation_path)
        self.vessels_buffer = vessels_buffer

    @staticmethod
    def _fix_classes(label):
        labels = {
            'Blood vessels': 'blood_vessels',
            'Inflammation': 'inflammations',
            'Endocarium': 'endocariums',
            'Fatty tissue': 'fatty_tissues'
        }

        return labels[label]

    def _load_annotation(self, annotation_path):
        files = glob.glob(f'{annotation_path}\\*.geojson')
        annotation_gdf = gpd.GeoDataFrame()

        for file in files:
            file_name = file.replace('/', '\\').split('\\').pop().split('.')[0]
            gj = geojson.load(open(file))

            gdf = gpd.GeoDataFrame.from_features(gj['features'])
            gdf['file_name'] = file_name
            annotation_gdf = gpd.GeoDataFrame(
                pd.concat([annotation_gdf, gdf], ignore_index=True))

        # Change classification column
        annotation_gdf['classification'] = annotation_gdf.apply(
            lambda row: row['classification']['name'], axis=1)

        # Drop unnecessary columns
        annotation_gdf = annotation_gdf.drop(
            columns=['object_type', 'isLocked'])  # ['object_type', 'isLocked'] ['objectType]

        # Change classifications for Immune cells, Quilty and remove Fibrotic tissue
        annotation_gdf.loc[annotation_gdf['classification'] ==
                           'Immune cells', 'classification'] = 'Inflammation'
        annotation_gdf.loc[annotation_gdf['classification']
                           == 'Quilty', 'classification'] = 'Inflammation'
        annotation_gdf = annotation_gdf[~(
            annotation_gdf['classification'] == 'Fibrotic tissue')]

        # Change classification names
        annotation_gdf['classification'] = annotation_gdf.apply(
            lambda row: self._fix_classes(row['classification']),
            axis=1
        )

        self.annotation_gdf = annotation_gdf

    @staticmethod
    def _create_polygon(x, y, size):
        coors = (x, y, x + size, y + size)

        return [
            [coors[0], coors[1]],
            [coors[2], coors[1]],
            [coors[2], coors[3]],
            [coors[0], coors[3]],
            [coors[0], coors[1]]
        ]

    @staticmethod
    def _check_result(annotations):
        is_empty = True
        for key, values in annotations.items():
            if len(values) > 0:
                is_empty = False
                break

        if is_empty:
            return 0
        else:
            return annotations

    def get_annotations_by_name(self, file_name, annotation_classes=None):
        if annotation_classes is None:
            annotation_classes = [
                'blood_vessels',
                'endocariums',
                'fatty_tissues',
                'inflammations'
            ]

        annotations = dict()
        for annotation_class in annotation_classes:
            class_annotations = self.annotation_gdf[
                (self.annotation_gdf['file_name'] == file_name)
                & (self.annotation_gdf['classification'] == annotation_class)
            ]

            if annotation_class == 'blood_vessels':
                annotations[f"{annotation_class}_patch"] = [
                    {'scaled_annotation': mapping(
                        row['geometry'].buffer(self.vessels_buffer))}
                    for index, row in class_annotations.iterrows()
                ]
            else:
                annotations[f"{annotation_class}_patch"] = [
                    {'scaled_annotation': mapping(row['geometry'])}
                    for index, row in class_annotations.iterrows()
                ]

        return self._check_result(annotations)

    def get_patch_annotations_by_name(self, file_name, patch, annotation_classes=None):
        if annotation_classes is None:
            annotation_classes = [
                'blood_vessels',
                'endocariums',
                'fatty_tissues',
                'inflammations'
            ]

        x, y, size = patch
        polygon = Polygon(self._create_polygon(x, y, size))

        annotations = dict()
        for annotation_class in annotation_classes:
            class_annotations = self.annotation_gdf[
                (self.annotation_gdf['file_name'] == file_name)
                & (self.annotation_gdf['classification'] == annotation_class)
                & (self.annotation_gdf.intersects(polygon))
            ]
            if annotation_class == 'blood_vessels':
                annotations[f"{annotation_class}_patch"] = [
                    {'scaled_annotation': mapping(
                        row['geometry'].buffer(self.vessels_buffer))}
                    for index, row in class_annotations.iterrows()
                ]
            else:
                annotations[f"{annotation_class}_patch"] = [
                    {'scaled_annotation': mapping(row['geometry'])}
                    for index, row in class_annotations.iterrows()
                ]

        return self._check_result(annotations)
