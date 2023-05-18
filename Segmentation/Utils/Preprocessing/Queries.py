def query_by_name(annotation_classes):
    queries = {
        "blood_vessels": """
            blood_vessels {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "endocariums": """
            endocariums {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "fatty_tissues": """
            fatty_tissues {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "fibrotic_tissues": """
            fibrotic_tissues {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "immune_cells": """
            immune_cells {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "inflammations": """
            inflammations {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "quilties": """
            quilties {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
    }

    final_queries = []
    for annotation_class in annotation_classes:
        final_queries.append(queries[annotation_class])

    return """
        query getAnnotations($file_name: String!, $xfactor: float8!, $yfactor: float8!) {
          files(where: {file_name: {_eq: $file_name}}) {
            file_name
            """ + ''.join(final_queries) + """
          }
        }
    """


def query_patch_by_name(annotation_classes):
    queries = {
        "blood_vessels": """
            blood_vessels_patch(args: {square: $square, xfactor: $xfactor, yfactor: $yfactor}) {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "endocariums": """
            endocariums_patch(args: {square: $square, xfactor: $xfactor, yfactor: $yfactor}) {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "fatty_tissues": """
            fatty_tissues_patch(args: {square: $square, xfactor: $xfactor, yfactor: $yfactor}) {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "fibrotic_tissues": """
            fibrotic_tissues_patch(args: {square: $square, xfactor: $xfactor, yfactor: $yfactor}) {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "immune_cells": """
            immune_cells_patch(args: {square: $square, xfactor: $xfactor, yfactor: $yfactor}) {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "inflammations": """
            inflammations_patch(args: {square: $square, xfactor: $xfactor, yfactor: $yfactor}) {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
        "quilties": """
            quilties_patch(args: {square: $square, xfactor: $xfactor, yfactor: $yfactor}) {
              scaled_annotation(args: {xfactor: $xfactor, yfactor: $yfactor})
            }
        """,
    }

    final_queries = []
    for annotation_class in annotation_classes:
        final_queries.append(queries[annotation_class])

    return """
        query getPatchAnnotations($file_name: String!, $square: geometry!, $xfactor: float8!, $yfactor: float8!) {
          files(where: {file_name: {_eq: $file_name}}) {
            file_name
            """ + ''.join(final_queries) + """
          }
        }
    """


def query_scaled_by_name(query):
    return """
                query getScaledAnnotationsWithName($file_name: String!, $xfactor: float8!, $yfactor: float8!) {
                    """ + query + """(args: {xfactor: $xfactor, yfactor: $yfactor}, where: {$file_name: {_eq: $file_name}}) {
                        annotation
                        file_name
                      }
                    }

            """


def query_scaled(query):
    return """
                query getScaledAnnotations($xfactor: float8!, $yfactor: float8!) {
                    """ + query + """(args: {xfactor: $xfactor, yfactor: $yfactor}) {
                        annotation
                        file_name
                      }
                    }

            """


def insert_one(query):
    return """
                mutation insertAnnotation($annotation: geometry!, $file_name: String!) {
                  """ + query + """(object: {annotation: $annotation, file_name: $file_name}) {
                    annotation
                    file_name
                    id
                  }
                }
            """


def insert_multiple(table_name):
    return """
           mutation insertAnnotations($objects: [""" + table_name + """_insert_input!]!) {
              insert_""" + table_name + """(objects: $objects) {
                returning {
                  annotation
                  file_name
                }
              }
            }
           """


def delete_rows(table_name):
    return """
                mutation deleteRows {
                  delete_""" + table_name + """(where: {}) {
                    returning {
                      id
                    }
                  }
                }
           """


def insert_file():
    return """
           mutation MyMutation($file_name: String!) {
              insert_files_one(object: {file_name: $file_name}) {
                file_name
                id
              }
            }
           """
