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

    for word, freq in fdist.most_common(200):  # top 200 most common
        other_words.append(word)

    # print(other_words)
    STOPWORDS.extend(other_words)

    # executionTime = (time.time() - startTime)
    # print('Execution time in seconds: ' + str(executionTime))
    return STOPWORDS
