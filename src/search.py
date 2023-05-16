import os
import re
import time
import requests
import pathlib
import numpy as np
import pandas as pd
import sqlite3
import shelve

import jaro
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch

from lupyne import engine
import lucene
from org.apache.lucene import queryparser, search
from org.apache.lucene.index import IndexWriterConfig, IndexWriter, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.analysis import Analyzer, StopwordAnalyzerBase
from org.apache.lucene.analysis.standard import StandardTokenizer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer, LowerCaseFilter, WhitespaceTokenizer, StopFilter, \
    StopAnalyzer
from org.apache.lucene.analysis.en import PorterStemFilter
from org.apache.lucene.document import Document, Field, TextField
from org.apache.pylucene.analysis import PythonAnalyzer
from google.cloud import storage, bigquery
from java.nio.file import Paths
from java.util import Arrays, HashSet


class SearchNotAvailable(Exception):
    pass


class ScorerNotAvailable(Exception):
    pass


class VectorizerNotAvailable(Exception):
    pass


def set_up_search(method, PATH_ontology, list_of_ontologies):
    index_objects = {}
    if method == 'pylucene':
        try:
            print("Loading PyLucene Indices...")
            for each in list_of_ontologies:
                path = os.path.join(os.path.join(PATH_ontology, each))
                myindexer = SearchPyLucene(each, path)
                index_objects[each] = myindexer
            print("Done.\n")
        except OSError:
            raise OSError("Indices not found.")
    elif method == 'elastic':
        try:
            print("If you haven't, please run generate_elastic_index.py after creating local ElasticSearch cluster.")
            print("Loading Elastic Search Indices...")
            for each in list_of_ontologies:
                myindexer = SearchElastic(each)
                myindexer.prepare_search()
                index_objects[each] = myindexer
            print("Done.\n")
        except:
            raise SearchNotAvailable('Please check your search engine.')
    elif method == 'tf-idf':
        try:
            print("Loading TF-IDF Index...")
            with shelve.open(os.path.join(PATH_ontology, 'tf_idf.shlv'), protocol=5) as shlv:
                index_objects['ngrams'] = shlv['ngrams']
                index_objects['vectorizer'] = shlv['model']
                index_objects['tf_idf_matrix'] = shlv['tf_idf_matrix']
            print("Done.\n")
        except FileNotFoundError:
            raise FileNotFoundError("Vectorizer shelve files not found, TF-IDF is not available.")
    return index_objects


def search_ontology(query, df_ontology, method, **kwargs):
    if method in ['jaro_winkler', 'partial_ratio']:
        searcher = SearchSimilarity(data=df_ontology)
        return searcher.get_search_results(query=query, scorer=method, **kwargs)
    elif method == 'tf_idf':  # NLP: tf_idf
        searcher = SearchTF_IDF(
            data=df_ontology,
            vectorizer=kwargs['indexes']['vectorizer'],
            tf_idf_matrix=kwargs['indexes']['tf_idf_matrix']
        )
        return searcher.get_search_results(query=query, **kwargs)
    elif method == 'UMLS':
        searcher = SearchUMLS()
        result = searcher.get_search_results(query=query)
        df_result = pd.DataFrame.from_records(result, columns=['LABEL', 'CODE'])
        df_result[method] = np.array([100] * len(df_result['CODE']))
        df_result['id'] = np.arange(len(df_result['CODE']))
        col_id = df_result.pop('id')
        df_result.insert(0, col_id.name, col_id)
        return df_result
    elif method == 'fts5':
        query = re.sub(r'[^A-Za-z0-9 ]+', '', query)
        tokens = re.split('\W+', query)
        query_tokens = ' OR '.join(tokens)
        if kwargs['triggered_id'] == 'search-btn.n_clicks':
            if kwargs['search_string'] != '':
                query = kwargs['search_string']
        df_result = kwargs['sql_searcher'].search_ontology_by_label(f'\"{query}\"')

        if df_result.empty:
            if kwargs['triggered_id'] == 'search-btn.n_clicks':
                df_result = kwargs['sql_searcher'].search_ontology_by_label(f'{query}')
            else:
                df_result = kwargs['sql_searcher'].search_ontology_by_label(f'{query_tokens}')
                query = query_tokens
            if df_result.empty:
                return query
        match_scores = [round(100 / len(df_result['CODE']) * i) for i in range(len(df_result['CODE']))]
        match_scores = match_scores[::-1]
        df_result[method] = match_scores
        df_result['id'] = np.arange(len(df_result['CODE']))
        col_id = df_result.pop('id')
        df_result.insert(0, col_id.name, col_id)
        return df_result
    elif method in ['pylucene', 'elastic']:
        if kwargs['triggered_id'] == 'search-btn.n_clicks' or kwargs['search_string'] in kwargs['listed_options']:
            query = kwargs['search_string']
        try:
            df_result = kwargs['indexes'][kwargs['ontology_filter']].get_search_results(query=query, **kwargs)
        except:
            query = re.sub(r'[^A-Za-z0-9 ]+', '', query)
            df_result = kwargs['indexes'][kwargs['ontology_filter']].get_search_results(query=query, **kwargs)

        if df_result.empty:
            return query
        df_result['id'] = np.arange(len(df_result['CODE']))
        col_id = df_result.pop('id')
        df_result.insert(0, col_id.name, col_id)
        return df_result
    else:
        return 'Invalid Search'


class SearchSQLite:
    def __init__(self, ontology, path_to_db):
        self.ontology = ontology
        self.path = path_to_db
        if '.appspot.com' in self.path:
            BUCKET_NAME = os.environ['BUCKET_NAME']
            ontology_relpath = os.path.relpath(self.path, BUCKET_NAME)
            BUCKET_PATH = os.path.join('ontology', ontology_relpath)
            head, tail = os.path.split(self.path)
            DATABASE_NAME_IN_RUNTIME = f"/tmp/{tail}"  # Remember that only the /tmp folder is writable within the directory

            storage_client = storage.Client(project=os.environ['PROJECT_ID'])
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(BUCKET_PATH)
            blob.download_to_filename(DATABASE_NAME_IN_RUNTIME)

            self.path = DATABASE_NAME_IN_RUNTIME

    def search_ontology_by_label(self, query):
        conn = sqlite3.connect(self.path)
        df_result = pd.read_sql(f"SELECT * FROM {self.ontology} WHERE LABEL MATCH '{query}' ORDER BY rank", conn)
        return df_result

    def search_ontology_by_code(self, query):
        conn = sqlite3.connect(self.path)
        df_result = pd.read_sql(f"SELECT * FROM {self.ontology} WHERE CODE MATCH '{query}'", conn)
        return df_result

    def get_all_ontology_no_data(self):
        conn = sqlite3.connect(self.path)
        df_result = pd.read_sql(f"SELECT CODE, LABEL FROM {self.ontology}", conn)
        df_result = df_result.reset_index().rename(columns={"index": "id"})
        return df_result

    def get_all_ontology_with_data(self):
        conn = sqlite3.connect(self.path)
        df_result = pd.read_sql(f"SELECT * FROM {self.ontology}", conn)
        df_result = df_result.reset_index().rename(columns={"index": "id"})
        return df_result


class PorterStemmerAnalyzer(PythonAnalyzer):
    def __init__(self):
        PythonAnalyzer.__init__(self)

    #  Override java function
    def createComponents(self, fieldName):
        source = StandardTokenizer()
        result = LowerCaseFilter(source)
        result = PorterStemFilter(result)
        # stopfilter: https://lucene.apache.org/core/8_11_0/core/index.html
        # result = StopFilter(result, StopwordAnalyzerBase.stopwords)
        return Analyzer.TokenStreamComponents(source, result)


class SearchPyLucene:
    def __init__(self, ontology, path):
        self.ontology = ontology
        self.path = path
        if '.appspot.com' in self.path:
            BUCKET_NAME = os.environ['BUCKET_NAME']
            ontology_relpath = os.path.relpath(self.path, BUCKET_NAME)
            BUCKET_PATH = os.path.join('ontology', ontology_relpath)
            INDEX_LOCATION_IN_RUNTIME = f"/tmp/{self.ontology}/"  # Remember that only the /tmp folder is writable within the directory
            if not os.path.isdir(INDEX_LOCATION_IN_RUNTIME):
                os.makedirs(INDEX_LOCATION_IN_RUNTIME)

            storage_client = storage.Client(project=os.environ['PROJECT_ID'])
            bucket = storage_client.get_bucket(BUCKET_NAME)
            blobs = bucket.list_blobs(prefix=BUCKET_PATH)
            for blob in blobs:
                if all(x not in blob.name for x in ['.db', '.csv']):
                    head, tail = os.path.split(blob.name)
                    FILE_PATH_IN_RUNTIME = os.path.join(INDEX_LOCATION_IN_RUNTIME, tail)
                    blob.download_to_filename(FILE_PATH_IN_RUNTIME)

            self.path = INDEX_LOCATION_IN_RUNTIME
        try:
            lucene.initVM()
        except:
            lucene.getVMEnv().attachCurrentThread()
        self.analyzer = PorterStemmerAnalyzer()
        self.config = IndexWriterConfig(self.analyzer)
        self.store = SimpleFSDirectory(Paths.get(self.path))

    def get_search_results(self, query, n=50, **kwargs):
        lucene.getVMEnv().attachCurrentThread()
        ireader = DirectoryReader.open(self.store)
        isearcher = search.IndexSearcher(ireader)

        # Parse a simple query that searches in label col:
        parser = queryparser.classic.QueryParser('LABEL', self.analyzer)
        query = parser.parse(query)
        scoreDocs = isearcher.search(query, n).scoreDocs
        hits = [isearcher.doc(scoreDoc.doc) for scoreDoc in scoreDocs]
        hits_list = []
        for i, hit in enumerate(hits):
            table = dict((field.name(), field.stringValue()) for field in hit.getFields())
            entry = {each_col: hit[each_col] for each_col in table.keys()}
            entry['pylucene'] = round(scoreDocs[i].score, 1)
            hits_list.append(entry)
        try:
            results = pd.DataFrame(hits_list, columns=list(hits_list[0].keys()))
            for each_col in results.columns:
                results.loc[results[each_col] == 'None', each_col] = np.nan
        except IndexError:
            results = pd.DataFrame()
        return results


# not used currently
# class SearchLupyne:
#     def __init__(self, results=None, indexer=None):
#         self.results = results
#         self.indexer = indexer
#
#     def prepareEngine(self):
#         lucene.initVM()
#
#     def createIndex(self, df, path):
#         self.indexer = engine.Indexer(path)
#         for each_col in df.columns:
#             if each_col == 'LABEL':
#                 self.indexer.set('LABEL', engine.Field.Text, stored=True)
#             else:
#                 self.indexer.set(each_col, stored=True)
#         for index, each_row in df.iterrows():
#             # implement way to add each col containing additional info for ontology code
#             self.indexer.add(CODE=each_row['CODE'], LABEL=each_row['LABEL'])
#         self.indexer.commit()
#
#     def loadIndex(self, path):
#         self.indexer = engine.Indexer(path)
#
#     def executeSearch(self, query):
#         lucene.getVMEnv().attachCurrentThread()
#         hits = self.indexer.search(f'LABEL: {query}')
#         hits_list = []
#         for i, hit in enumerate(hits):
#             entry = {each_col: hit[each_col] for each_col in hit.keys()}
#             entry['pylucene'] = round(hit.score, 1)
#             hits_list.append(entry)
#         try:
#             self.results = pd.DataFrame(hits_list, columns=list(hits_list[0].keys()))
#             for each_col in self.results.columns:
#                 self.results.loc[self.results[each_col] == 'None', each_col] = np.nan
#         except IndexError:
#             self.results = pd.DataFrame()


class SearchElastic:
    def __init__(self, ontology):
        self.connection = None
        self.es = None
        self.index = ontology

    def prepare_search(self):
        self.connection = "http://localhost:9200"
        self.es = Elasticsearch(self.connection)
        return

    def get_search_results(self, query, n=50, **kwargs):
        try:
            response = self.es.search(
                index=self.index,
                query={
                    "query_string": {
                        "query": query,
                    },
                },
                size=f'{n}',
            )
        except:
            raise SearchNotAvailable('Please check your search engine.')
        df_result = pd.DataFrame.from_records(response.body['hits']['hits'], columns=["_score", "_source"])
        df_source = pd.json_normalize(df_result["_source"])
        df_source["elastic"] = df_result["_score"].round(2)
        return df_source


class SearchTF_IDF:
    def __init__(self, data, vectorizer, tf_idf_matrix):
        self.data = data
        self.vectorizer = vectorizer
        self.matrix = tf_idf_matrix

    def get_search_results(self, query, n=50, **kwargs):
        if not self.vectorizer:
            raise VectorizerNotAvailable("Please define a vectorizer in the function call.")
        fitted_query = self.vectorizer.transform([query])
        scores = cosine_similarity(self.matrix, fitted_query)
        self.data['tf_idf'] = scores
        self.data = self.data.sort_values(by=['tf_idf'], ascending=False)
        return self.data[1:n + 1]


class SearchSimilarity:
    def __init__(self, data):
        self.data = data
        self.choices = list(self.data['LABEL'])

    def get_search_results(self, query, scorer, n=50, **kwargs):
        if scorer == 'jaro_winkler':
            score_func = jaro.jaro_winkler_metric
        elif scorer == 'partial_ratio':
            score_func = fuzz.partial_ratio
        else:
            raise ScorerNotAvailable()
        related = process.extractBests(query, self.choices, scorer=score_func, limit=n)
        df_related_score = pd.DataFrame(related[1:], columns=['LABEL', scorer])
        self.data = self.data[self.data['LABEL'].isin([i[0] for i in related[1:]])]
        self.data = self.data.merge(df_related_score, on='LABEL')
        self.data = self.data.sort_values(by=[scorer], ascending=False)
        return self.data


class SearchUMLS:
    def __init__(self):
        self.base_uri = 'https://uts-ws.nlm.nih.gov'
        self.api_key = os.getenv("UMLS_API_KEY")

    def get_search_results(self, query):
        path = '/search/current/'
        query = {'apiKey': self.api_key, 'string': query, 'sabs': 'SNOMEDCT_US', 'returnIdType': 'code'}
        output = requests.get(self.base_uri + path, params=query)
        outputJson = output.json()
        results = (([outputJson['result']])[0])['results']
        related = [(item['name'], item['ui']) for item in results]
        return related


def ngrams(string: str, n=10) -> list:
    """
    Takes an input string, cleans it and converts to ngrams.
    :param string: str
    :param n: int
    :return: list
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


def partial_ratio(string_1: str, string_2: str) -> float:
    """
    Calculates the fuzzywuzzy partial ratio between 2 strings.
    :param string_1: str
    :param string_2: str
    :return: float
    """
    ratio = fuzz.partial_ratio(string_1.lower(), string_2.lower())
    return ratio


def jaro_winkler(string_1: str, string_2: str) -> float:
    """
    Calculates the Jaro-Winkler score between 2 strings.
    :param string_1: str
    :param string_2: str
    :return: float
    """
    score = jaro.jaro_winkler_metric(string_1.lower(), string_2.lower())
    return score
