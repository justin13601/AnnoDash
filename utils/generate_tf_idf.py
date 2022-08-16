# REQUIRES config.yaml FROM generate_config.py.
# REQUIRES PREPROCESSED ONTOLOGY FILES FROM preprocess_ontology.py -> WIP.

import re
import os
import csv
import time
import yaml
import json
import errno
import shelve

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from related_ontologies.related import ngrams


def load_config(file):
    print(f'Loading {file}...')
    with open(file, "r") as f:
        configurations = yaml.unsafe_load(f)
        print('Done.\n')
        return configurations


# run
if __name__ == "__main__":
    config_file = '../config.yaml'
    if os.path.exists(config_file):
        print('Configuration file found.')
        config = load_config('../config.yaml')
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)

    PATH_base = '../'
    PATH_ontology = os.path.join(PATH_base, config.ontology.location)
    PATH_data = os.path.join(PATH_base, config.directories.data)
    df_ontology = pd.read_csv(os.path.join(PATH_ontology, 'LoincTableCore.csv'), dtype=object)
    df_ontology = df_ontology[df_ontology['CLASSTYPE'] == str(1)]
    df_ontology.drop(df_ontology[df_ontology.STATUS != 'ACTIVE'].index, inplace=True)
    df_ontology.drop(['CLASSTYPE', 'STATUS', 'EXTERNAL_COPYRIGHT_NOTICE', 'VersionFirstReleased', 'VersionLastChanged'],
                     axis=1,
                     inplace=True)
    print(f"Ontology codes (CLASSTYPE=1, Laboratory Terms Class) loaded and processed.\n")

    ontology_dict = pd.Series(df_ontology.LONG_COMMON_NAME.values, index=df_ontology.LOINC_NUM.values).to_dict()
    df_ontology_new = pd.DataFrame(
        {'LOINC_NUM': list(ontology_dict.keys()), 'LONG_COMMON_NAME': list(ontology_dict.values())})
    df_ontology_new = df_ontology_new.reset_index().rename(columns={"index": "id"})

    t1 = time.time()
    ontology_names = list(df_ontology_new['LONG_COMMON_NAME'].unique())

    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(ontology_names)

    t = time.time() - t1
    print("Time:", t)
    print(tf_idf_matrix.shape)

    # save using shelve
    with shelve.open('tf_idf.shlv', protocol=5) as shlv:
        shlv['ngrams'] = ngrams
        shlv['model'] = vectorizer
        shlv['tf_idf_matrix'] = tf_idf_matrix

    # save elementary idf and vocab
    # np.savetxt('idf_.txt', vectorizer.idf_)
    # json.dump(vectorizer.vocabulary_, open('vocabulary.json', mode='w'))

    # PICKLE
    # save matrix
    # pickle.dump(tf_idf_matrix, open("ontology_tf_idf_matrix_n=10.pkl", "wb"))

    # save vectorizer
    # pickle.dump(vectorizer, open("ontology_vectorizer_n=10.pkl", "wb"))

    # save vectorizer vocabulary
    # pickle.dump(vectorizer.vocabulary_, open("ontology_vectorizer_vocabulary_n=10.pkl", "wb"))

    # upload matrix
    # tf_idf_matrix = pickle.load(open("ontology_tf_idf_matrix_n=10.pkl","rb"))

    # upload vectorizer
    # vectorizer = pickle.load(open("ontology_vectorizer_n=10.pkl", "rb"))

    # upload vectorizer
    # vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, vocabulary=pickle.load(
    #     open(os.path.join(PATH_data, 'ontology_vectorizer_vocabulary_n=10.pkl'), "rb")))
    # vectorizer.fit_transform(list(df_ontology_new['LONG_COMMON_NAME'].unique()))
    # tf_idf_matrix = pickle.load(open(os.path.join(PATH_data, 'ontology_tf_idf_matrix_n=10.pkl'), "rb"))
