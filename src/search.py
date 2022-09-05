import sqlite3
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

    def closeSearch(self):
        self.conn.close()
