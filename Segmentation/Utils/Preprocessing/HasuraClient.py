import requests

from .Queries import query_scaled_by_name, query_scaled, insert_file, insert_multiple, query_by_name, query_patch_by_name


class HasuraClient:
    def __init__(self, url, header):
        self.url = url
        self.header = header

    def get_blood_vessels_by_name(self, file_name, xfactor=1, yfactor=1):
        result = self._run_query(
            query_scaled_by_name("get_blood_vessels_scaled"),
            {"file_name": file_name, "xfactor": xfactor, "yfactor": yfactor}
        )

        if 'errors' in result:
            print(result)
            return None

        return result['data']['get_blood_vessels_scaled']

    def get_blood_vessels(self, xfactor=1, yfactor=1):
        result = self._run_query(
            query_scaled("get_blood_vessels_scaled"),
            {"xfactor": xfactor, "yfactor": yfactor}
        )

        if 'errors' in result:
            print(result)
            return None

        return result['data']['get_blood_vessels_scaled']

    def get_fatty_tissues_by_name(self, file_name, xfactor=1, yfactor=1):
        result = self._run_query(
            query_scaled_by_name("get_fatty_tissues_scaled"),
            {"file_name": file_name, "xfactor": xfactor, "yfactor": yfactor}
        )

        if 'errors' in result:
            print(result)
            return None

        return result['data']['get_fatty_tissues_scaled']

    def get_fatty_tissues(self, xfactor=1, yfactor=1):
        result = self._run_query(
            query_scaled("get_fatty_tissues_scaled"),
            {"xfactor": xfactor, "yfactor": yfactor}
        )

        if 'errors' in result:
            print(result)
            return None

        return result['data']['get_fatty_tissues_scaled']

    def get_endocariums_by_name(self, file_name, xfactor=1, yfactor=1):
        result = self._run_query(
            query_scaled_by_name("get_endocariums_scaled"),
            {"file_name": file_name, "xfactor": xfactor, "yfactor": yfactor}
        )

        if 'errors' in result:
            print(result)
            return None

        return result['data']['get_endocariums_scaled']

    def get_endocariums(self, xfactor=1, yfactor=1):
        result = self._run_query(
            query_scaled("get_endocariums_scaled"),
            {"xfactor": xfactor, "yfactor": yfactor}
        )

        if 'errors' in result:
            print(result)
            return None

        return result['data']['get_endocariums_scaled']

    def get_inflammations_by_name(self, file_name, xfactor=1, yfactor=1):
        result = self._run_query(
            query_scaled_by_name("get_inflammations_scaled"),
            {"file_name": file_name, "xfactor": xfactor, "yfactor": yfactor}
        )

        if 'errors' in result:
            print(result)
            return None

        return result['data']['get_inflammations_scaled']

    def get_inflammations(self, xfactor=1, yfactor=1):
        result = self._run_query(
            query_scaled("get_inflammations_scaled"),
            {"xfactor": xfactor, "yfactor": yfactor}
        )

        if 'errors' in result:
            print(result)
            return None

        return result['data']['get_inflammations_scaled']

    def import_annotations_to_DB(self, json_data, file_name):
        result = self._run_query(
            insert_file(),
            {"file_name": file_name}
        )

        if 'errors' in result:
            print(result)
            return

        data = {
            'blood_vessels': [],
            'fatty_tissues': [],
            'inflammations': [],
            'endocariums': [],
            'fibrotic_tissues': [],
            'quilties': [],
            'immune_cells': []
        }

        for feature in json_data['features']:
            class_type = feature['properties']['classification']['name']
            geometry = feature['geometry']

            if class_type == 'Blood vessels':
                data['blood_vessels'].append({
                    "annotation": geometry,
                    "file_name": file_name
                })
            elif class_type == 'Fatty tissue':
                data['fatty_tissues'].append({
                    "annotation": geometry,
                    "file_name": file_name
                })
            elif class_type == 'Inflammation':
                data['inflammations'].append({
                    "annotation": geometry,
                    "file_name": file_name
                })
            elif class_type == 'Endocarium':
                data['endocariums'].append({
                    "annotation": geometry,
                    "file_name": file_name
                })
            elif class_type == 'Fibrotic tissue':
                data['fibrotic_tissues'].append({
                    "annotation": geometry,
                    "file_name": file_name
                })
            elif class_type == 'Quilty':
                data['quilties'].append({
                    "annotation": geometry,
                    "file_name": file_name
                })
            elif class_type == 'Immune cells':
                data['immune_cells'].append({
                    "annotation": geometry,
                    "file_name": file_name
                })

        for key in data:
            result = self._run_query(
                insert_multiple(key),
                {'objects': data[key]}
            )

            if 'errors' in result:
                print(result)
                return

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

        result = self._run_query(
            query_by_name(annotation_classes),
            {
                "file_name": file_name,
                "xfactor": xfactor,
                "yfactor": yfactor}
        )

        return self._handle_files(result)

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
        patch_square = self._create_polygon(x, y, size)

        result = self._run_query(
            query_patch_by_name(annotation_classes),
            {
                "file_name": file_name,
                "xfactor": xfactor,
                "yfactor": yfactor,
                "square": patch_square
            }
        )

        return self._handle_files(result)

    def _run_query(self, query, variables):
        request = requests.post(
            self.url,
            headers=self.header,
            json={"query": query, "variables": variables},
        )
        assert request.ok, f"Failed with code {request.status_code}"
        return request.json()

    @staticmethod
    def _handle_files(result):
        if 'errors' in result:
            print(result)
            return None

        if len(result['data']['files']) == 0:
            return 0

        return result['data']['files'][0]

    @staticmethod
    def _create_polygon(x, y, size):
        coors = (x, y, x + size, y + size)

        return {
            "type": "Polygon",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:EPSG::4326"
                }
            },
            "coordinates": [
                [
                    [coors[0], coors[1]],
                    [coors[2], coors[1]],
                    [coors[2], coors[3]],
                    [coors[0], coors[3]],
                    [coors[0], coors[1]]
                ]
            ]
        }
