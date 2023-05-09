import yaml
from ml_collections import config_dict

# configurations
save_directory = 'results-json/chartevents'
concepts_file = 'demo-data/demo_chartevents_user_1.csv'
data_file = 'demo-data/CHARTEVENTS.csv'
ontology_directory = 'kind-lab.appspot.com'
ontology_search_method = 'pylucene'  # [sqlite, pylucene, tf-idf]
gpt_support = True
'''
UMLS_API_KEY = 'please load your API key from an environment variable or secret management service if ontology_search_method -> umls'
OPENAI_API_KEY = 'please load your API key from an environment variable or secret management service if gpt_support -> True
'''

graph_title_font = 'verdana'
graph_title_color = 'Black'
graph_title_size = 25
graph_body_font = 'verdana'
graph_body_color = 'Black'
graph_body_size = 12
graph_show_spikes = True


def generateConfig():
    cfg = config_dict.ConfigDict()
    cfg.results = save_directory
    cfg.concepts = concepts_file
    cfg.data = data_file
    cfg.ontology = config_dict.ConfigDict()
    cfg.ontology.location = ontology_directory
    cfg.ontology.search = ontology_search_method
    cfg.ontology.gpt = gpt_support

    cfg.kwargs = config_dict.ConfigDict()
    cfg.kwargs.title_font = graph_title_font
    cfg.kwargs.title_color = graph_title_color
    cfg.kwargs.title_size = graph_title_size
    cfg.kwargs.text_font = graph_body_font
    cfg.kwargs.text_color = graph_body_color
    cfg.kwargs.text_size = graph_body_size
    cfg.kwargs.spikes = graph_show_spikes

    app_labels_dictionary = {
        'app': 'mimic-iv-dash',
        'version': 2.0
    }
    metadata_dictionary = {
        'name': 'development',
        'labels': config_dict.FrozenConfigDict(app_labels_dictionary)
    }
    cfg.metadata = config_dict.FrozenConfigDict(metadata_dictionary)

    with open('./app/config.yaml', 'w') as yaml_file:
        yaml.dump(cfg, yaml_file)
    return cfg


if __name__ == '__main__':
    config = generateConfig()
    print(config)
