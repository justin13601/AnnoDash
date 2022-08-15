import yaml
from ml_collections import config_dict


def load_config(file):
    print(f'Loading {file}...')
    with open(file, "r") as f:
        configurations = yaml.safe_load(f)
        print('Done.\n')
        return configurations


cfg = config_dict.ConfigDict()

cfg.directories = config_dict.ConfigDict()
cfg.directories.results = 'results-json'
cfg.directories.concepts = config_dict.ConfigDict()
cfg.directories.concepts.location = 'demo-data'
cfg.directories.concepts.filename = 'D_LABITEMS.csv'
cfg.directories.data = config_dict.ConfigDict()
cfg.directories.data.location = 'demo-data'
cfg.directories.data.filename = 'LABEVENTS.csv'

cfg.ontology = config_dict.ConfigDict()
cfg.ontology.location = 'ontology'
cfg.ontology.filename = 'LoincClassType_1.csv'
cfg.ontology.related = config_dict.ConfigDict()
cfg.ontology.related.scorer = 'tf_idf'
cfg.ontology.related.location = config_dict.placeholder(str)
cfg.ontology.related.location = 'ontology'

cfg.graphs = config_dict.ConfigDict()
cfg.graphs.pairs = config_dict.ConfigDict()
cfg.graphs.pairs.pair_one = config_dict.ConfigDict()
cfg.graphs.pairs.pair_one.item_1 = 50821
cfg.graphs.pairs.pair_one.item_2 = 50818
cfg.graphs.pairs.pair_one.label = 'Blood Gas'
cfg.graphs.pairs.pair_two = config_dict.ConfigDict()
cfg.graphs.pairs.pair_two.item_1 = 50912
cfg.graphs.pairs.pair_two.item_2 = 50971
cfg.graphs.pairs.pair_two.label = 'Chemistry'
cfg.graphs.pairs.pair_three = config_dict.ConfigDict()
cfg.graphs.pairs.pair_three.item_1 = 51222
cfg.graphs.pairs.pair_three.item_2 = 51300
cfg.graphs.pairs.pair_three.label = 'Complete Blood Count'
cfg.graphs.kwargs = config_dict.ConfigDict()
cfg.graphs.kwargs.title_font = 'verdana'
cfg.graphs.kwargs.title_color = 'Black'
cfg.graphs.kwargs.title_size = 25
cfg.graphs.kwargs.text_font = 'verdana'
cfg.graphs.kwargs.text_color = 'Black'
cfg.graphs.kwargs.text_size = 12
cfg.graphs.kwargs.height = 375
cfg.graphs.kwargs.spikes = True

########################################

cfg.temp = config_dict.ConfigDict()
cfg.temp.five_percent_dataset = config_dict.placeholder(bool)
cfg.temp.five_percent_dataset = True
cfg.temp.mimic_iv_version = config_dict.placeholder(float)
cfg.temp.mimic_iv_version = 2.0

########################################
# labels_dictionary = {
#     'app': 'dash-web'
# }
#
# metadata_dictionary = {
#     'name': 'mimic-iv-dash-development',
#     'labels': config_dict.FrozenConfigDict(labels_dictionary)
# }
#
# cfg.metadata = config_dict.FrozenConfigDict(metadata_dictionary)
#
# temp_dictionary = {
#     'test': True,
#     'mimic_iv_version': 2.0
# }
#
# cfg.temp = config_dict.FrozenConfigDict(temp_dictionary)
########################################


with open('../config.yaml', 'w') as yaml_file:
    yaml.dump(cfg, yaml_file)

print(cfg)
