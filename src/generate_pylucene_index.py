import os
import time
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

from src.search import SearchSQLite


def query_ontology(ontology):
    database_file = f'{ontology}.db'
    path = os.path.join(os.path.join('../ontology', ontology), database_file)

    mysearch = SearchSQLite()
    mysearch.prepareSearch(path)
    mysearch.getOntologyFull(ontology)
    mysearch.closeSearch()
    df_ontology = mysearch.df
    del mysearch
    return df_ontology


class PorterStemmerAnalyzer(PythonAnalyzer):
    def __init__(self):
        PythonAnalyzer.__init__(self)

    def createComponents(self, fieldName):
        source = StandardTokenizer()
        result = LowerCaseFilter(source)
        result = PorterStemFilter(result)
        return Analyzer.TokenStreamComponents(source, result)


ontology_path = "../ontology"
load = False

startTime = time.time()
######################################################################
# LOINC
######################################################################

df_loinc = query_ontology('loinc')
lucene.initVM()

analyzer = PorterStemmerAnalyzer()
config = IndexWriterConfig(analyzer)
store = SimpleFSDirectory(Paths.get(os.path.join(ontology_path, 'loinc')))

if not load:
    writer = IndexWriter(store, config)
    for i, each_row in df_loinc.iterrows():
        doc = Document()
        for each_col in df_loinc.columns:
            try:
                doc.add(Field(each_col, each_row[each_col], TextField.TYPE_STORED))
            except:
                doc.add(Field(each_col, str(each_row[each_col]), TextField.TYPE_STORED))
        writer.addDocument(doc)
    writer.close()

# search the index:
ireader = DirectoryReader.open(store)
isearcher = search.IndexSearcher(ireader)

# Parse a simple query that searches for "tests":
parser = queryparser.classic.QueryParser('LABEL', analyzer)
query = parser.parse('tests')
print(query)
hits = isearcher.search(query, len(df_loinc.index)).scoreDocs
print(len(hits))  # 820

# Iterate through the results:
for hit in hits:
    hitDoc = isearcher.doc(hit.doc)
    print(hitDoc)

ireader.close()
store.close()

######################################################################
# SNOMED
######################################################################

df_snomed = query_ontology('snomed')
lucene.initVM()

analyzer = PorterStemmerAnalyzer()
config = IndexWriterConfig(analyzer)
store = SimpleFSDirectory(Paths.get(os.path.join(ontology_path, 'snomed')))

if not load:
    writer = IndexWriter(store, config)
    for i, each_row in df_snomed.iterrows():
        doc = Document()
        for each_col in df_snomed.columns:
            try:
                doc.add(Field(each_col, each_row[each_col], TextField.TYPE_STORED))
            except:
                doc.add(Field(each_col, str(each_row[each_col]), TextField.TYPE_STORED))
        writer.addDocument(doc)
    writer.close()

# search the index:
ireader = DirectoryReader.open(store)
isearcher = search.IndexSearcher(ireader)

# Parse a simple query that searches for "tests":
parser = queryparser.classic.QueryParser('LABEL', analyzer)
query = parser.parse('tests')
print(query)
hits = isearcher.search(query, len(df_loinc.index)).scoreDocs
print(len(hits))  # 2913

# # Iterate through the results:
# for hit in hits:
#     hitDoc = isearcher.doc(hit.doc)
#     print(hitDoc['LABEL'])

ireader.close()
store.close()

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
