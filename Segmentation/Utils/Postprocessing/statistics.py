import argparse
import geopandas
import geojson
import cv2 as cv
import os
import pandas as pd
import json
import string
import random


class Statistics:
    def __init__(self, output_dir, class_names=None):
        self.output_dir = output_dir
        if class_names is None:
            self.class_names = {
                'Blood vessels': 'Blood vessels',
                'Endocardium': 'Endocarium',
                'Inflammation': 'Inflammation',
                'Tissue': 'Region*',
            }
        else:
            self.class_names = class_names

    def _get_id(self, row):
        letters = string.ascii_lowercase + string.digits

        final_id = ''.join(random.choice(letters) for i in range(8)) + '-' + ''.join(random.choice(letters) for i in range(4)) + '-' + ''.join(random.choice(letters)
                                                                                                                                               for i in range(4)) + '-' + ''.join(random.choice(letters) for i in range(4)) + '-' + ''.join(random.choice(letters) for i in range(12))

        return final_id

    def _handle_id(self, geojson_file):
        for index, feature in enumerate(geojson_file['features']):
            geojson_file['features'][index]['id'] = feature['properties']['id']
            del geojson_file['features'][index]['properties']['id']

        return geojson_file

    def calculate_area(self, gpd_annotations, class_name):
        overall_area = 0
        areas = []
        ids = []
        for index, row in gpd_annotations.iterrows():
            if row['classification']['name'] == class_name:
                if class_name == 'Region*' and row.geometry.type == 'MultiPolygon':
                    for polygon in row.geometry:
                        areas.append(polygon.area)
                        overall_area += polygon.area
                else:
                    areas.append(row.geometry.area)
                    ids.append(gpd_annotations.iloc[index]['id'])
                    gpd_annotations.iloc[index]['measurements'].append({
                        "name": "Area [px^2]",
                        "value": row.geometry.area
                    })
                    overall_area += row.geometry.area
        return overall_area, areas, len(areas), ids

    def calculate_infla_area(self, gpd_annotations, out_dir):
        overall_area, areas, count, ids = self.calculate_area(
            gpd_annotations, self.class_names['Inflammation'])

        return overall_area, areas, count, ids

    def calculate_endocard_area(self, gpd_annotations, out_dir):
        overall_area, areas, _, ids = self.calculate_area(
            gpd_annotations, self.class_names['Endocardium'])

        return overall_area, areas, ids

    def calculate_tissue_area(self, gpd_cell_annotations):
        overall_area, _, _, _ = self.calculate_area(
            gpd_cell_annotations, self.class_names['Tissue'])

        return overall_area

    def calculate_mean_infla_area(self, infla_area, infla_count, tissue_area):
        mean_area = infla_area / infla_count
        mean_area_percent = infla_area / tissue_area * 100

        return mean_area, mean_area_percent

    def calculate_mean_endocard_area(self, endocard_area, tissue_area):
        area_percent = endocard_area / tissue_area * 100

        return area_percent

    def calculate_cell_statistics(self, gpd_cell_annotations, gpd_annotations, out_dir):
        cell_counts = []
        immune_cells_counts = []
        ids = []
        for index, row in gpd_annotations.iterrows():
            if row['classification']['name'] == self.class_names['Inflammation']:
                annotation = row.geometry
                cell_annotations = gpd_cell_annotations[
                    gpd_cell_annotations.geometry.intersects(annotation)
                ]

                cell_count = len(cell_annotations)
                ids.append(gpd_annotations.iloc[index]['id'])
                immune_cells = []
                for _, cell in cell_annotations.iterrows():
                    if cell['classification']['name'] == 'Immune cells':
                        immune_cells.append(cell)

                immune_cells_count = len(immune_cells)
                cell_counts.append(cell_count)
                immune_cells_counts.append(immune_cells_count)
                gpd_annotations.iloc[index]['measurements'].append({
                    "name": "Cell count",
                    "value": cell_count
                })
                gpd_annotations.iloc[index]['measurements'].append({
                    "name": "Immune cell count",
                    "value": immune_cells_count
                })

        df = pd.DataFrame({
            "Type": "Inflammation",
            "Id": ids,
            "Cell count": cell_counts,
            "Immune cell count": immune_cells_counts
        })
        df.to_csv(
            f'{out_dir}/inflammation_cell_statistics.csv', index=False)

    def run_statistics(self, cell_annotations, predicted_annotations):
        file_name = predicted_annotations.replace(
            '\\', '/').split('/')[-1].split('.')[0]
        out_dir = f'{self.output_dir}/{file_name}'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        gpd_cell_annotations = geopandas.GeoDataFrame.from_file(
            cell_annotations)
        gpd_annotations = geopandas.GeoDataFrame.from_file(
            predicted_annotations)
        gpd_annotations['id'] = gpd_annotations['classification'].apply(
            self._get_id)

        def get_props(row):
            return []

        gpd_annotations['measurements'] = gpd_annotations['classification'].apply(
            get_props)

        infla_area, infla_areas, infla_count, infla_ids = self.calculate_infla_area(
            gpd_annotations, out_dir)
        endocard_area, endocard_areas, endocard_ids = self.calculate_endocard_area(
            gpd_annotations, out_dir)
        tissue_area = self.calculate_tissue_area(gpd_cell_annotations)

        mean_infla_area, mean_infla_area_percent = self.calculate_mean_infla_area(
            infla_area, infla_count, tissue_area)

        endocard_area_percent = self.calculate_mean_endocard_area(
            endocard_area, tissue_area)

        self.calculate_cell_statistics(
            gpd_cell_annotations, gpd_annotations, out_dir)

        names = [f'Inflammation' for i in range(
            len(infla_areas))] + [f'Endocard' for i in range(len(endocard_areas))]
        areas = [area for area in infla_areas] + \
            [area for area in endocard_areas]
        ids = [i_id for i_id in infla_ids] + \
            [e_id for e_id in endocard_ids]
        df = pd.DataFrame({
            "Type": names,
            "Id": ids,
            "Area [px^2]": areas
        })
        df.to_csv(f'{out_dir}/statistics.csv', index=False)

        df = pd.DataFrame({
            "Overall Inflammation Area [px^2]": [infla_area],
            "Number of Inflammations [px^2]": [infla_count],
            "Overall Endocardium Area [px^2]": [endocard_area],
            "Overall tissue Area [px^2]": [tissue_area],
            "Mean Area of Inflammation [px^2]": [mean_infla_area],
            "Percentage of Inflammation [%]": [mean_infla_area_percent],
            "Percentage of Endocardium [%]": [endocard_area_percent]
        })
        df.to_csv(f'{out_dir}/overall_statistics.csv', index=False)
        #gpd_annotations.to_file(f'{out_dir}/{file_name}.geojson', driver='GeoJSON')
        geojson_file = gpd_annotations.to_json()
        geojson_file = self._handle_id(json.loads(geojson_file))
        with open(f'{out_dir}/{file_name}.geojson', "w") as outfile:
            json.dump(geojson_file, outfile)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_annotations', type=str,
                        help='Path to cell annotations')
    parser.add_argument('--predicted_annotations', type=str,
                        help='Path to predicted annotations')
    parser.add_argument('--output_dir', type=str,
                        help='Path to output directory')

    args = parser.parse_args()

    Statistics(args.output_dir).run_statistics(
        args.cell_annotations, args.predicted_annotations)
