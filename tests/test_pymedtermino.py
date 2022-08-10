# import sqlite3
#
# conn = sqlite3.connect(':memory:')
# cur = conn.cursor()
# conn.enable_load_extension(True)
#
# for (val,) in cur.execute('pragma compile_options'):
#     print(val)


import os
import numpy as np
import pandas as pd
import pymedtermino
from pymedtermino.snomedct import *

pymedtermino.LANGUAGE = "en"
pymedtermino.REMOVE_SUPPRESSED_CONCEPTS = True

concept = SNOMEDCT[251890007]
print(concept)
print(concept.terms)
print(concept.parents)
print(concept.children)

for ancestor in concept.ancestors():
    print(ancestor)

observable_identity = 363787002
laboratory_test = 15220000
print(concept.is_a(SNOMEDCT[observable_identity]))
print(concept.is_part_of(SNOMEDCT[observable_identity]))

# for concept in SNOMEDCT.all_concepts():
#     print(concept)


file_name = 'sct2_Concept_Snapshot_INT_20220731.txt'
snomed_dir = r'C:\Users\Justin\Documents\SickKids\MIMIC\NLM UMLS\SNOMED-CT_InternationalEdition\SnomedCT_InternationalRF2_PRODUCTION_20220731T120000Z\Snapshot\Terminology'

path = os.path.join(snomed_dir, file_name)
df_snomed = pd.read_csv(path, sep='\t')
print(df_snomed.shape)

df_snomed = df_snomed.loc[df_snomed['active'] == 1]
print(df_snomed.shape)
df_snomed = df_snomed.sort_values('effectiveTime').drop_duplicates('id', keep='last')

ontology_sub = observable_identity


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


df_snomed['observable_entity'] = df_snomed['id'].apply(is_part_of_class)
df_snomed_new = df_snomed.loc[df_snomed['observable_entity'] == True]
df_snomed_new.drop(columns=['observable_entity'])
print(df_snomed_new.shape)

df_snomed_new['label'] = df_snomed_new['id'].apply(get_term_from_code)
print(df_snomed_new.shape)

label_column = df_snomed_new.pop('label')
df_snomed_new.insert(1, 'label', label_column)

df_snomed_new.to_csv('SNOMED_CT_Observable_Entity.csv', index=False)
