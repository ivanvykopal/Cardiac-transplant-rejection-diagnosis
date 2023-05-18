import json
import psycopg2
from psycopg2.extras import execute_values
from shapely import wkb
from shapely.geometry import Polygon, mapping


class Database:
    def __init__(self, config_name=None):
        if config_name is None:
            config_name = './Utils/Preprocessing/database_config.json'

        self.cursor = None
        self.connection = None

        with open(config_name, 'r') as config_file:
            self.config = json.load(config_file)

        self._create_connection()

    def _create_connection(self):
        try:
            print(f"Connecting to database: '{self.config['database']}'")
            self.connection = psycopg2.connect(**self.config)
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        print(f"Connection created!")

    def disconnect(self):
        if self.cursor:
            self.cursor.close()

        self.connection.close()
        print(f"Database connection closed!")

    def get_connection(self):
        return self.connection

    def get_cursor(self):
        if self.cursor is None:
            self.cursor = self.connection.cursor()
        return self.cursor

    def execute_insert_many(self, insert_query, values):
        if self.cursor is None:
            self.get_cursor()

        execute_values(self.cursor, insert_query, values)

    def execute_insert(self, insert_query, value):
        if self.cursor is None:
            self.get_cursor()

        self.cursor.execute(insert_query, value)

    def execute_select(self, select_query):
        if self.cursor is None:
            self.get_cursor()

        try:
            self.cursor.execute(select_query)
        except Exception:
            self._create_connection()
            self.cursor.execute(select_query)

        return self.cursor.fetchall()

    def commit(self):
        self.connection.commit()

    def import_annotations(self, json_data, file_name):
        self.execute_insert(
            "INSERT INTO files (file_name) VALUES(%s)", (file_name, ))

        data = {
            'immune_cells': [],
            'muscles': [],
            'others': []
        }

        self.commit()

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

    def get_annotations_by_table(self, table_name, file, xfactor, yfactor):
        if table_name == 'blood_vessels':
            select = f'SELECT ST_Buffer(ST_Scale(annotation, {xfactor}, {yfactor}), 15)'
        else:
            select = f'SELECT ST_Scale(annotation, {xfactor}, {yfactor})'

        try:
            return self.execute_select(f"{select} FROM {table_name} WHERE file_name='{file}';")
        except psycopg2.OperationalError:
            self._create_connection()
            return self.execute_select(f"{select} FROM {table_name} WHERE file_name='{file}';")

    def get_annotations_patch_by_table(self, table_name, file, square, xfactor, yfactor):
        if table_name == 'blood_vessels':
            select = f'SELECT ST_Buffer(ST_Scale(annotation, {xfactor}, {yfactor}), 15)'
        else:
            select = f'SELECT ST_Scale(annotation, {xfactor}, {yfactor})'

        try:
            return self.execute_select(f"{select} FROM {table_name} WHERE file_name='{file}' AND ST_Intersects(ST_Scale(annotation, {xfactor}, {yfactor}), ST_Scale('{square}', {xfactor}, {yfactor}));")
        except psycopg2.OperationalError:
            self._create_connection()
            return self.execute_select(f"{select} FROM {table_name} WHERE file_name='{file}';")

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

    def get_annotations_by_name(self, file_name, annotation_classes=None, xfactor=1, yfactor=1):
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

        annotations = dict()
        for annotation_class in annotation_classes:
            class_annotations = self.get_annotations_by_table(
                annotation_class, file_name, xfactor, yfactor)
            annotations[annotation_class] = [
                {'scaled_annotation': mapping(
                    wkb.loads(annotation[0], hex=True))}
                for annotation in class_annotations
            ]

        return self._check_result(annotations)

    def get_patch_annotations_by_name(self, file_name, patch, annotation_classes=None, xfactor=1, yfactor=1):
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

        x, y, size = patch
        hex_square = wkb.dumps(
            Polygon(self._create_polygon(x, y, size)), hex=True, srid=4326)

        annotations = dict()
        for annotation_class in annotation_classes:
            class_annotations = self.get_annotations_patch_by_table(
                annotation_class, file_name, hex_square, xfactor, yfactor)
            annotations[f"{annotation_class}_patch"] = [
                {'scaled_annotation': mapping(
                    wkb.loads(annotation[0], hex=True))}
                for annotation in class_annotations
            ]

        return self._check_result(annotations)


if __name__ == '__main__':
    database = Database()
    data = database.execute_select("SELECT * FROM blood_vessels LIMIT 10")
    geom = wkb.loads(data[0][2], hex=True)
    print(geom)
