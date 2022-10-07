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

from java.nio.file import Paths


class SearchSQLite:
    def __init__(self, conn=None, df=None):
        self.conn = conn
        self.df = df

    def prepareSearch(self, path_to_db):
        self.conn = sqlite3.connect(path_to_db)
        c = self.conn.cursor()

    def searchOntologyLabel(self, ontology, query):
        self.df = pd.read_sql(f"SELECT * FROM {ontology} WHERE LABEL MATCH '{query}' ORDER BY rank", self.conn)

    def searchOntologyCode(self, ontology, query):
        self.df = pd.read_sql(f"SELECT * FROM {ontology} WHERE CODE MATCH '{query}'", self.conn)

    def getOntology(self, ontology):
        self.df = pd.read_sql(f"SELECT CODE, LABEL FROM {ontology}", self.conn)
        self.df = self.df.reset_index().rename(columns={"index": "id"})

    def getOntologyFull(self, ontology):
        self.df = pd.read_sql(f"SELECT * FROM {ontology}", self.conn)
        self.df = self.df.reset_index().rename(columns={"index": "id"})

    def closeSearch(self):
        self.conn.close()


class PorterStemmerAnalyzer(PythonAnalyzer):
    def __init__(self):
        PythonAnalyzer.__init__(self)

    def createComponents(self, fieldName):
        source = StandardTokenizer()
        result = LowerCaseFilter(source)
        result = PorterStemFilter(result)
        return Analyzer.TokenStreamComponents(source, result)


class SearchPyLucene:
    def __init__(self, results=None, analyzer=None, config=None, store=None):
        self.results = results
        self.analyzer = analyzer
        self.config = config
        self.store = store

    def prepareEngine(self):
        lucene.initVM()
        self.analyzer = PorterStemmerAnalyzer()
        self.config = IndexWriterConfig(self.analyzer)

    def loadIndex(self, path):
        self.store = SimpleFSDirectory(Paths.get(path))

    # createIndex not implemented

    def executeSearch(self, query):
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


class SearchLupyne:
    def __init__(self, results=None, indexer=None):
        self.results = results
        self.indexer = indexer

    def prepareEngine(self):
        lucene.initVM()

    def createIndex(self, df, path):
        self.indexer = engine.Indexer(path)
        for each_col in df.columns:
            if each_col == 'LABEL':
                self.indexer.set('LABEL', engine.Field.Text, stored=True)
            else:
                self.indexer.set(each_col, stored=True)
        for index, each_row in df.iterrows():
            # implement way to add each col containing additional info for ontology code
            self.indexer.add(CODE=each_row['CODE'], LABEL=each_row['LABEL'])
        self.indexer.commit()

    def loadIndex(self, path):
        self.indexer = engine.Indexer(path)

    def executeSearch(self, query):
        lucene.getVMEnv().attachCurrentThread()
        hits = self.indexer.search(f'LABEL: {query}')
        hits_list = []
        for i, hit in enumerate(hits):
            entry = {each_col: hit[each_col] for each_col in hit.keys()}
            entry['pylucene'] = round(hit.score, 1)
            hits_list.append(entry)
        try:
            self.results = pd.DataFrame(hits_list, columns=list(hits_list[0].keys()))
            for each_col in self.results.columns:
                self.results.loc[self.results[each_col] == 'None', each_col] = np.nan
        except IndexError:
            self.results = pd.DataFrame()


class SearchTF_IDF:
    def __init__(self, ngram=None, vectorizer=None, matrix=None):
        self.ngram = ngram
        self.vectorizer = vectorizer
        self.matrix = matrix
