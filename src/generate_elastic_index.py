import os
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.search import SearchSQLite

from src.stopwords import *

ontology_path = "../ontology"


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
    path = os.path.join(os.path.join(ontology_path, ontology), database_file)

    mysearch = SearchSQLite(ontology, path)
    df_ontology = mysearch.get_all_ontology_with_data()
    return df_ontology


STOPWORDS = get_stopwords()
ontologies = list_available_ontologies()

connection = "http://localhost:9200"
es = Elasticsearch(connection)

startTime = time.time()

for each_ontology in ontologies:
    print(f"Generating Elasticsearch index for {each_ontology}...")
    df = query_ontology(each_ontology)

    properties = {}
    for col in list(df.columns):
        properties[col] = {"type": "text", "analyzer": "extended_snowball_analyzer"}
    mappings = {
        "properties": properties
    }

    settings = {
        'analysis': {
            'analyzer': {
                'extended_snowball_analyzer': {
                    'type': 'snowball',
                    'stopwords': STOPWORDS,
                },
            },
        },
    }

    es.indices.create(index=each_ontology, mappings=mappings, settings=settings)

    bulk_data = []
    for i, row in df.iterrows():
        source = {}
        for col in list(df.columns):
            source[col] = row[col]
        bulk_data.append(
            {
                "_index": "loinc",
                "_id": i,
                "_source": source
            }
        )
    bulk(es, bulk_data)
    es.indices.refresh(index=each_ontology)
    print(es.cat.count(index=each_ontology, format="json"))

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
