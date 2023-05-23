import os
import time
import lucene
from org.apache.lucene import queryparser, search
from org.apache.lucene.index import IndexWriterConfig, IndexWriter, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.analysis import Analyzer, StopwordAnalyzerBase
from org.apache.lucene.analysis.standard import StandardTokenizer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer, LowerCaseFilter, WhitespaceTokenizer, StopFilter
from org.apache.lucene.analysis.en import PorterStemFilter
from org.apache.lucene.document import Document, Field, TextField
from org.apache.pylucene.analysis import PythonAnalyzer

from java.nio.file import Paths
from java.util import Arrays, HashSet

from src.search import SearchSQLite


class InvalidOntology(Exception):
    pass


def list_available_ontologies():
    print(f'Loading available ontologies...')
    if '.appspot.com' in ontology_path:
        ontologies = ['loinc', 'snomed']
    else:
        directory_contents = os.listdir(ontology_path)
        ontologies = [item for item in directory_contents if os.path.isdir(os.path.join(ontology_path, item))]
    if not ontologies:
        raise InvalidOntology
    print(f"{', '.join(each_ontology.upper() for each_ontology in ontologies)} codes available.\n")
    return ontologies


def query_ontology(ontology):
    database_file = f'{ontology}.db'
    path = os.path.join(os.path.join('../ontology', ontology), database_file)

    mysearch = SearchSQLite(ontology, path)
    df_ontology = mysearch.get_all_ontology_with_data()
    return df_ontology


class PorterStemmerAnalyzer(PythonAnalyzer):
    def __init__(self):
        PythonAnalyzer.__init__(self)

    @staticmethod
    def get_stops(result):
        myStops = ['a', 'b', 'c']
        stop_set = HashSet()
        for stopWord in myStops:
            stop_set.add(stopWord)
        return stop_set

    def createComponents(self, fieldName):
        source = StandardTokenizer()
        result = LowerCaseFilter(source)
        result = PorterStemFilter(result)
        # result = StopFilter(result, StopwordAnalyzerBase.stopwords)
        return Analyzer.TokenStreamComponents(source, result)


ontology_path = "../ontology"
load = False
ontologies = list_available_ontologies()

startTime = time.time()

for each_ontology in ontologies:
    print(f"Generating index for {each_ontology}...")
    df = query_ontology(each_ontology)
    lucene.initVM()

    analyzer = PorterStemmerAnalyzer()
    config = IndexWriterConfig(analyzer)
    store = SimpleFSDirectory(Paths.get(os.path.join(ontology_path, each_ontology)))

    if not load:
        writer = IndexWriter(store, config)
        for i, each_row in df.iterrows():
            doc = Document()
            for each_col in df.columns:
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
    print("Querying:", query)
    hits = isearcher.search(query, len(df.index)).scoreDocs
    print("Results found:", len(hits))

    # Iterate through the results:
    for i, hit in enumerate(hits):
        hitDoc = isearcher.doc(hit.doc)
        print(hitDoc)
        if i > 5:
            break

    ireader.close()
    store.close()
    print('\n')

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
