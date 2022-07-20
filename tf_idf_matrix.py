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


# set ngram N:
def ngrams(string, n=10):
    """
    Takes an input string, cleans it and converts to ngrams.
    """
    string = str(string)
    string = string.lower()  # lower case
    string = fix_text(string)  # fix text
    string = string.encode("ascii", errors="ignore").decode()  # remove non ascii chars
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'", "-"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'  # remove punc, brackets etc...
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.title()  # normalise case - capital at start of each word
    string = re.sub(' +', ' ', string).strip()  # get rid of multiple spaces and replace with a single
    string = ' ' + string + ' '  # pad names for ngrams...
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def load_config(file):
    print(f'Loading {file}...')
    with open(file, "r") as f:
        configurations = yaml.safe_load(f)
        print('Done.\n')
        return configurations


# run
if __name__ == "__main__":
    config_file = 'config.yaml'
    if os.path.exists(config_file):
        print('Configuration file found.')
        config = load_config('config.yaml')
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)

    PATH_loinc = config['loinc']['location']
    PATH_data = config['directories']['data']
    df_loinc = pd.read_csv(os.path.join(PATH_loinc, 'LoincTableCore.csv'), dtype=object)
    df_loinc = df_loinc[df_loinc['CLASSTYPE'] == str(1)]
    df_loinc.drop(df_loinc[df_loinc.STATUS != 'ACTIVE'].index, inplace=True)
    df_loinc.drop(['CLASSTYPE', 'STATUS', 'EXTERNAL_COPYRIGHT_NOTICE', 'VersionFirstReleased', 'VersionLastChanged'],
                  axis=1,
                  inplace=True)
    print(f"LOINC codes (CLASSTYPE=1, Laboratory Terms Class) loaded and processed.\n")

    loinc_dict = pd.Series(df_loinc.LONG_COMMON_NAME.values, index=df_loinc.LOINC_NUM.values).to_dict()
    df_loinc_new = pd.DataFrame(
        {'LOINC_NUM': list(loinc_dict.keys()), 'LONG_COMMON_NAME': list(loinc_dict.values())})
    df_loinc_new = df_loinc_new.reset_index().rename(columns={"index": "id"})

    t1 = time.time()
    loinc_names = list(df_loinc_new['LONG_COMMON_NAME'].unique())

    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(loinc_names)

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
    # pickle.dump(tf_idf_matrix, open("LOINC_tf_idf_matrix_n=10.pkl", "wb"))

    # save vectorizer
    # pickle.dump(vectorizer, open("LOINC_vectorizer_n=10.pkl", "wb"))

    # save vectorizer vocabulary
    # pickle.dump(vectorizer.vocabulary_, open("LOINC_vectorizer_vocabulary_n=10.pkl", "wb"))

    # upload matrix
    # tf_idf_matrix = pickle.load(open("LOINC_tf_idf_matrix_n=10.pkl","rb"))

    # upload vectorizer
    # vectorizer = pickle.load(open("LOINC_vectorizer_n=10.pkl", "rb"))

    # upload vectorizer
    # vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, vocabulary=pickle.load(
    #     open(os.path.join(PATH_data, 'LOINC_vectorizer_vocabulary_n=10.pkl'), "rb")))
    # vectorizer.fit_transform(list(df_loinc_new['LONG_COMMON_NAME'].unique()))
    # tf_idf_matrix = pickle.load(open(os.path.join(PATH_data, 'LOINC_tf_idf_matrix_n=10.pkl'), "rb"))
