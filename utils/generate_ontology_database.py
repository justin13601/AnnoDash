import os
import re
import numpy as np
import pandas as pd
import sqlite3
import pymedtermino
from pymedtermino import *
from pymedtermino.snomedct import *

# import sqlite_spellfix

######################################################################
# LOINC
######################################################################

conn = sqlite3.connect('../ontology/loinc.db')
conn.enable_load_extension(True)
# conn.load_extension(sqlite_spellfix.extension_path())
c = conn.cursor()

# Create table with headers
c.execute('''CREATE VIRTUAL TABLE loinc using fts5
             (CODE, LABEL, COMPONENT, PROPERTY,
             TIME_ASPCT, SYSTEM, SCALE_TYP,
             METHOD_TYP, CLASS, CLASSTYPE, SHORTNAME, ONTOLOGY)''')
conn.commit()

file_name = 'LoincTableCore.csv'
dir = r'C:\Users\Justin\PycharmProjects\mimic-iv-dash\ontology'
path = os.path.join(dir, file_name)

# Convert CSV to SQL
df_loinc = pd.read_csv(path, low_memory=False)
df_loinc.rename(columns={'LOINC_NUM': 'CODE', 'LONG_COMMON_NAME': 'LABEL'}, inplace=True)
df_loinc['ONTOLOGY'] = 'LOINC Core 2.72'
df_loinc.drop(df_loinc[df_loinc.STATUS != 'ACTIVE'].index, inplace=True)
df_loinc.drop(['STATUS', 'EXTERNAL_COPYRIGHT_NOTICE', 'VersionFirstReleased', 'VersionLastChanged'],
              axis=1, inplace=True)
df_loinc.to_sql('loinc', conn, if_exists='append', index=False)

######################################################################
# SNOMED-CT
######################################################################

conn = sqlite3.connect('../ontology/snomed.db')
conn.enable_load_extension(True)
# conn.load_extension(sqlite_spellfix.extension_path())
c = conn.cursor()

# Create table with headers
c.execute('''CREATE VIRTUAL TABLE snomed using fts5
             (CODE, LABEL, EFFECTIVE_TIME, PARENTS, CHILDREN, HIERARCHY, ONTOLOGY)''')
conn.commit()

pymedtermino.LANGUAGE = "en"
pymedtermino.REMOVE_SUPPRESSED_CONCEPTS = True

file_name = 'sct2_Concept_Snapshot_INT_20220731.txt'
dir = r'C:\Users\Justin\PycharmProjects\mimic-iv-dash\ontology'
path = os.path.join(dir, file_name)

df_snomed = pd.read_csv(path, sep='\t')
df_snomed = df_snomed.loc[df_snomed['active'] == 1]
df_snomed = df_snomed.sort_values('effectiveTime').drop_duplicates('id', keep='last')
df_snomed.drop(['active', 'moduleId', 'definitionStatusId'],
               axis=1, inplace=True)
df_snomed['ONTOLOGY'] = 'SNOMED-CT International 07/31/2022'


def apply_and_concat(dataframe, field, func, column_names):
    return pd.concat((
        dataframe,
        dataframe[field].apply(
            lambda cell: pd.Series(func(cell), index=column_names))), axis=1)


hierarchy_not_found = []


def get_term_data(code):
    global hierarchy_not_found
    try:
        term = SNOMEDCT[code].term
        LABEL = re.sub("[\(\[].*?[\)\]]", "", term)
        LABEL = LABEL.strip()
        num_parents = len(SNOMEDCT[code].parents)
        num_children = len(SNOMEDCT[code].children)

        snomed_hierarchies = {
            'Clinical finding': ['finding', 'disorder'],
            'Procedure': ['procedure', 'regime/therapy'],
            'Event': ['event'],
            'Observable entity': ['observable entity', 'function'],
            'Situation with explicit context': ['situation'],
            'Pharmaceutical / biologic product': ['product', 'medicinal product form', 'medicinal product',
                                                  'clinical drug'],
            'Social context': ['social concept', 'person', 'ethnic group', 'racial group', 'religion/philosophy',
                               'occupation', 'life style', 'family'],
            'Body structure': ['body structure', 'morphologic abnormality', 'cell', 'cell structure'],
            'Organism': ['organism'],
            'Physical object': ['physical object'],
            'Substance': ['substance'],
            'Specimen': ['specimen'],
            'Physical Force': ['physical force'],
            'Environment or geographical location': ['environment', 'geographic location', 'environment / location'],
            'Staging and scales': ['assessment scale', 'tumor staging', 'staging scale'],
            'Qualifier value': ['qualifier value', 'basic dose form', 'disposition', 'administration method', 'role',
                                'intended site', 'release characteristic', 'transformation', 'supplier', 'dose form',
                                'state of matter', 'unit of presentation', 'product name'],
            'SNOMED CT Model Component': ['attribute', 'link assertion', 'core metadata concept', 'metadata',
                                          'foundation metadata concept', 'linkage concept', 'namespace concept',
                                          'OWL metadata concept'],
            'Special concept': ['inactive concept', 'navigational concept'],
            'Record artifact': ['record artifact']
        }
        hierarchy = term[term.rfind('(') + 1:term.rfind(')')]
        try:
            first_level = [k for k, v in snomed_hierarchies.items() if hierarchy in v][0]
        except:
            hierarchy_not_found.append(code)
            print(f'Top-level hierarchy not found for {code}. Term: {SNOMEDCT[code].term}')
            return LABEL, num_parents, num_children, np.nan
        hierarchy_label = f'{first_level} -> {hierarchy}'
        return LABEL, num_parents, num_children, hierarchy_label
    except ValueError:
        # print(f'Suppressed: {code}.')
        return np.nan, np.nan, np.nan, np.nan


df_snomed.rename(columns={'id': 'CODE', 'effectiveTime': 'EFFECTIVE_TIME'}, inplace=True)
df_snomed = apply_and_concat(df_snomed, 'CODE', get_term_data, ['LABEL', 'PARENTS', 'CHILDREN', 'HIERARCHY'])
df_snomed = df_snomed[df_snomed['LABEL'].notna()]

df_snomed = df_snomed[['CODE', 'LABEL', 'EFFECTIVE_TIME', 'PARENTS', 'CHILDREN', 'HIERARCHY', 'ONTOLOGY']]

df_snomed.to_sql('snomed', conn, if_exists='append', index=False)

######################################################################

# conn = sqlite3.connect("../ontology/loinc.db")
# c = conn.cursor()
# df = pd.read_sql("SELECT * FROM loinc WHERE label MATCH 'Total OR Calculated OR CO2' ORDER BY rank", conn)
# df_2 = pd.read_sql("SELECT * FROM loinc EXCEPT SELECT * FROM loinc WHERE label MATCH 'TOTAL OR Calculated OR CO2'", conn)
#
# df_3 = df.append(df_2, ignore_index=True)

# print(df_3)

# conn = sqlite3.connect("../ontology/snomed.db")
# c = conn.cursor()
# df = pd.read_sql("SELECT * FROM snomed WHERE label MATCH 'Total OR Calculated OR CO2' ORDER BY rank", conn)
# df_2 = pd.read_sql("SELECT * FROM snomed EXCEPT SELECT * FROM snomed WHERE label MATCH 'TOTAL OR Calculated OR CO2'", conn)
#
# df_3 = df.append(df_2, ignore_index=True)

# print(df_3)


print('Done.')
