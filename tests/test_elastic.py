import os
import time
from elasticsearch import Elasticsearch


def main():
    start_time = time.time()

    connection = "http://localhost:9200"
    es = Elasticsearch(connection)
    print(es.info().body)

    elapsed_time = time.time() - start_time
    print('--------Elastic Search Time:', elapsed_time, 'seconds--------')
    return


if __name__ == "__main__":
    main()
