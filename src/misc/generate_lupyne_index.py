import os
import time
import lucene
from lupyne import engine

from src.search import SearchSQLite

ontology_path = '../ontology'

load = False


# https://stackoverflow.com/questions/47668000/pylucene-indexer-and-retriever-sample


def query_ontology(ontology):
    database_file = f'{ontology}.db'
    path = os.path.join(os.path.join(ontology_path, ontology), database_file)

    mysearch = SearchSQLite()
    mysearch.prepareSearch(path)
    mysearch.getOntologyFull(ontology)
    mysearch.closeSearch()
    df_ontology = mysearch.df
    del mysearch
    return df_ontology


startTime = time.time()
######################################################################
# LOINC
######################################################################

df_loinc = query_ontology('loinc')
lucene.initVM()
storeDirectory = os.path.join(ontology_path, 'loinc')
indexer = engine.Indexer(storeDirectory)

if not load:
    indexer.set('CODE', stored=True)
    indexer.set('LABEL', engine.Field.Text, stored=True)
    indexer.set('SYSTEM', stored=True)
    indexer.set('SCALE_TYP', stored=True)
    indexer.set('METHOD_TYP', stored=True)
    indexer.set('CLASS', stored=True)
    for i, each in df_loinc.iterrows():
        indexer.add(CODE=each['CODE'], LABEL=each['LABEL'], SYSTEM=each['SYSTEM'],
                    SCALE_TYP=each['SCALE_TYP'], METHOD_TYP=str(each['METHOD_TYP']),
                    CLASS=each['CLASS'])
    indexer.commit()

hits = indexer.search('LABEL: test')
for hit in hits:
    print(hit.dict)
    print(f"{hit['CODE']} - {hit['LABEL']}: {hit.score}")
print(len(hits))  # 640

######################################################################
# SNOMED
######################################################################

df_snomed = query_ontology('snomed')
lucene.initVM()
storeDirectory = os.path.join(ontology_path, 'snomed')
indexer = engine.Indexer(storeDirectory)

if not load:
    indexer.set('CODE', stored=True)
    indexer.set('LABEL', engine.Field.Text, stored=True)
    indexer.set('EFFECTIVE_TIME', stored=True)
    indexer.set('HIERARCHY', stored=True)
    indexer.set('SEMANTIC_TAG', stored=True)
    for i, each in df_snomed.iterrows():
        indexer.add(CODE=each['CODE'], LABEL=each['LABEL'], EFFECTIVE_TIME=each['EFFECTIVE_TIME'],
                    HIERARCHY=each['HIERARCHY'], SEMANTIC_TAG=each['SEMANTIC_TAG'])
    indexer.commit()

hits = indexer.search('LABEL: test')
for hit in hits:
    print(hit.dict)
    print(f"{hit['CODE']} - {hit['LABEL']}: {hit.score}")
print(len(hits))  # 2661

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
