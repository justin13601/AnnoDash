import os
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.search import SearchSQLite


def query_ontology(ontology):
    database_file = f'{ontology}.db'
    path = os.path.join(os.path.join('../ontology', ontology), database_file)

    mysearch = SearchSQLite(ontology, path)
    df_ontology = mysearch.get_all_ontology_with_data()
    return df_ontology


ontology_path = "../ontology"
load = False

connection = "http://localhost:9200"
es = Elasticsearch(connection)

startTime = time.time()
######################################################################
# LOINC
######################################################################

df_loinc = query_ontology('loinc')

bulk_data = []
for i, row in df_loinc.iterrows():
    bulk_data.append(
        {
            "_index": "loinc",
            "_id": i,
            "_source": {
                "CODE": row["CODE"],
                "LABEL": row["LABEL"],
                "SYSTEM": row["SYSTEM"],
                "SCALE_TYP": row["SCALE_TYP"],
                "METHOD_TYP": row["METHOD_TYP"],
                "CLASS": row["CLASS"],
            }
        }
    )

bulk(es, bulk_data)
es.indices.refresh(index="loinc")
print(es.cat.count(index="loinc", format="json"))

######################################################################
# SNOMED
######################################################################

df_snomed = query_ontology('snomed')

bulk_data = []
for i, row in df_snomed.iterrows():
    bulk_data.append(
        {
            "_index": "snomed",
            "_id": i,
            "_source": {
                "CODE": row["CODE"],
                "LABEL": row["LABEL"],
                "EFFECTIVE_TIME": row["EFFECTIVE_TIME"],
                "HIERARCHY": row["HIERARCHY"],
                "SEMANTIC_TAG": row["SEMANTIC_TAG"],
            }
        }
    )

bulk(es, bulk_data)
es.indices.refresh(index="snomed")
print(es.cat.count(index="snomed", format="json"))

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
