import os
import pathlib
import numpy as np
import pandas as pd
import sqlite3
from lupyne import engine
import lucene
from org.apache.lucene import queryparser, search
from org.apache.lucene.index import IndexWriterConfig, IndexWriter, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.analysis.standard import StandardTokenizer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer, LowerCaseFilter, WhitespaceTokenizer, StopFilter, \
    StopAnalyzer
from org.apache.lucene.analysis.en import PorterStemFilter
from org.apache.lucene.document import Document, Field, TextField
from org.apache.pylucene.analysis import PythonAnalyzer
from google.cloud import storage, bigquery

from java.nio.file import Paths


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

            storage_client = storage.Client()
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

    def get_all_ontology(self):
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

    # camelCase to override java function?
    def createComponents(self, fieldName):
        source = StandardTokenizer()
        result = LowerCaseFilter(source)
        result = PorterStemFilter(result)
        return Analyzer.TokenStreamComponents(source, result)


class SearchPyLucene:
    def __init__(self, ontology, path):
        self.ontology = ontology
        self.path = os.path.join(os.path.join('ontology', path))
        # if '.appspot.com' in self.path:
        #     BUCKET_NAME = os.environ['BUCKET_NAME']
        #     ontology_relpath = os.path.relpath(self.path, BUCKET_NAME)
        #     BUCKET_PATH = os.path.join('ontology', ontology_relpath)
        #     head, tail = os.path.split(self.path)
        #     DATABASE_NAME_IN_RUNTIME = f"/tmp/{tail}"  # Remember that only the /tmp folder is writable within the directory
        #
        #     storage_client = storage.Client()
        #     bucket = storage_client.bucket(BUCKET_NAME)
        #     blob = bucket.blob(BUCKET_PATH)
        #     blob.download_to_filename(DATABASE_NAME_IN_RUNTIME)
        #
        #     self.path = DATABASE_NAME_IN_RUNTIME
        try:
            lucene.initVM()
        except:
            lucene.getVMEnv().attachCurrentThread()
        self.analyzer = PorterStemmerAnalyzer()
        self.config = IndexWriterConfig(self.analyzer)
        self.store = SimpleFSDirectory(Paths.get(self.path))
        self.results = pd.DataFrame()

    def execute_search(self, query):
        lucene.getVMEnv().attachCurrentThread()
        ireader = DirectoryReader.open(self.store)
        isearcher = search.IndexSearcher(ireader)

        # Parse a simple query that searches in label col:
        parser = queryparser.classic.QueryParser('LABEL', self.analyzer)
        query = parser.parse(query)
        scoreDocs = isearcher.search(query, 250).scoreDocs
        hits = [isearcher.doc(scoreDoc.doc) for scoreDoc in scoreDocs]
        hits_list = []
        for i, hit in enumerate(hits):
            table = dict((field.name(), field.stringValue()) for field in hit.getFields())
            entry = {each_col: hit[each_col] for each_col in table.keys()}
            entry['pylucene'] = round(scoreDocs[i].score, 1)
            hits_list.append(entry)
        try:
            self.results = pd.DataFrame(hits_list, columns=list(hits_list[0].keys()))
            for each_col in self.results.columns:
                self.results.loc[self.results[each_col] == 'None', each_col] = np.nan
        except IndexError:
            self.results = pd.DataFrame()

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
#
#
# class SearchTF_IDF:
#     def __init__(self, ngram=None, vectorizer=None, matrix=None):
#         self.ngram = ngram
#         self.vectorizer = vectorizer
#         self.matrix = matrix
