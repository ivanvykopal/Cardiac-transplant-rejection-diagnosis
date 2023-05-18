import argparse

parser = argparse.ArgumentParser(description='Identification of inflammations in VSI files.')

parser.add_argument(
    '-i',
    '--image_path',
    type=str,
    help='path to vsi file or directory with vsi files',
    required=False
)

parser.add_argument(
    '-g',
    '--geojson_path',
    type=str,
    help='path to geojson file or directory with geojson files'
)

parser.add_argument(
    '-o',
    '--output_path',
    type=str,
    help='path to output'
)

parser.add_argument(
    '-a',
    '--algorithm',
    type=str,
    help='algorithm to use',
    choices=['dbscan', 'dilatation'],
    default='dbscan'
)

parser.add_argument(
    '-c',
    '--num_cells',
    type=int,
    help='number of cells in inflammatory',
    required=False,
    default=15
)

parser.add_argument(
    '-d',
    '--cell_distance',
    type=int,
    help='max allowed distance between cells belonging into one cluster',
    required=False,
    default=100
)

parser.add_argument(
    '-m',
    '--min_area',
    type=int,
    help='minimum area of inflammatory',
    required=False,
    default=50_000
)

parser.add_argument(
    '--only_immune',
    type=bool,
    help='flag if to use only immune cells or not',
    default=False
)
