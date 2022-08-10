# Util script to preprocess ontologies
# Accepts LOINC .csv files or SNOMED-CT .txt files
# Saves preprocessed .csv file for input into config.yaml and MIMIC-Dash

import os
import numpy as np
import pandas as pd
import pymedtermino
from pymedtermino.snomedct import *

# Ontology (str): "loinc" or "snomed"
# Please note PyMedTermino is required for SNOMED-CT ontology filtering. Please follow installation instructions:
# https://pythonhosted.org/PyMedTermino/tuto_en.html#installation
ontology = "snomed"

# For LOINC, enter classtype (int) ex: 1
# For SNOMED, enter hierarchy level (int) ex: 363787002 (Observable Entity) or 15220000 (Laboratory Test)
ontology_sub = 15220000

# Ontology file/directory
# ex: 'LoincTableCore.csv' or 'sct2_Concept_Snapshot_INT_20220731.txt'
file_name = 'sct2_Concept_Snapshot_INT_20220731.txt'
dir = r'C:\Users\Justin\PycharmProjects\mimic-iv-dash\demo-data'
path = os.path.join(dir, file_name)


def is_part_of_class(code):
    try:
        return SNOMEDCT[code].is_a(SNOMEDCT[ontology_sub])
    except ValueError:
        return np.nan


def get_term_from_code(code):
    try:
        return SNOMEDCT[code].term
    except ValueError:
        return np.nan


class InvalidOntology(Exception):
    pass


class InvalidClassTypeLOINC(Exception):
    pass


class InvalidHierarchyLevelSNOMEDCT(Exception):
    pass


if ontology == "loinc":
    if ontology_sub not in [1, 2, 3, 4]:
        raise InvalidClassTypeLOINC
    print(f"Filtering LOINC codes for classtype: {ontology_sub}.")

    df_loinc = pd.read_csv(path, dtype=object)
    df_loinc = df_loinc[df_loinc['CLASSTYPE'] == str(ontology_sub)]
    df_loinc.drop(df_loinc[df_loinc.STATUS != 'ACTIVE'].index, inplace=True)
    df_loinc.drop(
        ['CLASSTYPE', 'STATUS', 'EXTERNAL_COPYRIGHT_NOTICE', 'VersionFirstReleased', 'VersionLastChanged'],
        axis=1,
        inplace=True)

    save_file = f'../LOINC_ClassType_{ontology_sub}.csv'
    df_loinc.to_csv(save_file, index=False)

elif ontology == "snomed":
    if not isinstance(ontology_sub, int):
        raise InvalidHierarchyLevelSNOMEDCT
    print(f"Filtering SNOMED-CT codes for: {ontology_sub} ({SNOMEDCT[ontology_sub].term}).")

    pymedtermino.LANGUAGE = "en"
    pymedtermino.REMOVE_SUPPRESSED_CONCEPTS = True

    df_snomed = pd.read_csv(path, sep='\t')
    df_snomed = df_snomed.loc[df_snomed['active'] == 1]
    df_snomed = df_snomed.sort_values('effectiveTime').drop_duplicates('id', keep='last')

    df_snomed['ontology_sub'] = df_snomed['id'].apply(is_part_of_class)
    df_snomed = df_snomed.loc[df_snomed['ontology_sub'] == True]
    df_snomed.drop(columns=['ontology_sub'])

    df_snomed['label'] = df_snomed['id'].apply(get_term_from_code)
    label_column = df_snomed.pop('label')
    df_snomed.insert(1, 'label', label_column)

    save_file = '../SNOMED_CT_Observable_Entity.csv'
    df_snomed.to_csv(save_file, index=False)
else:
    raise InvalidOntology

print(f"Done. Saved at {save_file}")
