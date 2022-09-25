import sqlite3
import lucene
import numpy as np
from lupyne import engine
import pandas as pd


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
