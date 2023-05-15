import os
import time

from src.search import SearchSQLite

import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist


def query_ontology(ontology):
    database_file = f'{ontology}.db'
    path = os.path.join(os.path.join('../ontology', ontology), database_file)

    mysearch = SearchSQLite(ontology, path)
    df_ontology = mysearch.get_all_ontology_with_data()
    return df_ontology


def get_stopwords():
    # startTime = time.time()

    STOPWORDS = stopwords.words('english')

    other_words = ['type']

    df_loinc = query_ontology('loinc')
    df_snomed = query_ontology('snomed')

    label_text = ' '.join(df_loinc['LABEL'].values.tolist())
    label_text += ' '.join(df_snomed['LABEL'].values.tolist())

    tokens = word_tokenize(label_text)
    fdist = FreqDist(tokens)  # frequency distribution of the tokens

    ontology_stopwords = fdist.most_common(200)  # top 200 most common
    exclude = ['Serum', 'Entire', 'Presence', 'Ab', 'Plasma', 'oral', 'dose', 'Blood', 'Urine', 'virus', 'artery',
               'joint', 'syndrome', 'antibody', 'neoplasm', 'spinal', 'IgG', 'skin', 'nerve', 'contrast', 'Genus',
               'bone', 'Tissue', 'tablet', 'hour', 'blood', 'tissue', 'type', 'Salmonella', 'past', 'acid', 'muscle',
               'disease', 'IgE', 'vein', 'bilateral', 'Titer', 'Congenital', 'eye', 'fracture', 'NAA', 'upper', 'care',
               'Cerebral', 'antigen', 'probe', 'milliliter', 'malignant', 'Open', 'limb', 'cell', 'Primary', 'disorder',
               'Family', 'device', 'Immunoassay', 'poisoning', 'gene', 'Ag', 'XR', 'finger', 'tendon', 'hand',
               'parenteral', 'panel', 'anterior', 'lesion', 'PROMIS', 'cells', 'therapy', 'foot', 'valve', 'injection',
               'ligament', 'tumor', 'IgM', 'injury', 'guidance', 'Molecular', 'IV', 'Susceptibility', 'posterior',
               'Immunofluorescence', 'genetics', 'DNA', 'level', 'Views', 'spine', 'Body', 'thoracic', 'surface',
               'Identifier', 'score', 'Acute', 'pressure', 'wound', 'cord', 'biopsy', 'toe', 'cutaneous',
               'immunoglobulin', 'gland', 'head', 'CMS', 'hydrochloride', 'capsule', 'overdose', 'Excision', 'vertebra',
               'Closed', 'Chronic', 'infection', 'lateral', 'tomography', 'assessment', 'stain']

    ontology_stopwords_new = [elem for elem in ontology_stopwords if elem in exclude]
    print(ontology_stopwords_new)

    STOPWORDS.extend(ontology_stopwords_new)
    STOPWORDS = list(map(lambda x: x.lower(), STOPWORDS))
    STOPWORDS = list(dict.fromkeys(STOPWORDS))  # remove duplicates

    # executionTime = (time.time() - startTime)
    # print('Execution time in seconds: ' + str(executionTime))
    # print(STOPWORDS)
    return STOPWORDS


get_stopwords()
