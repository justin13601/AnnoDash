import yaml
from ml_collections import config_dict

cfg = config_dict.ConfigDict()

cfg.directories = config_dict.ConfigDict()
cfg.directories.results = 'results-json/demo'
cfg.directories.concepts = config_dict.ConfigDict()
cfg.directories.concepts.location = 'demo-data'
cfg.directories.concepts.filename = 'demo_chartevents_user_1.csv'
cfg.directories.data = config_dict.ConfigDict()
cfg.directories.data.location = 'demo-data'
cfg.directories.data.filename = 'CHARTEVENTS.csv'

cfg.ontology = config_dict.ConfigDict()
cfg.ontology.location = 'ontology'
cfg.ontology.related = config_dict.ConfigDict()
cfg.ontology.related.location = config_dict.placeholder(str)
# cfg.ontology.related.location = 'ontology/related'
cfg.ontology.related.umls_apikey = config_dict.placeholder(str)
# cfg.ontology.related.umls_apikey = '1c3bfb9d-bcd1-472b-8ff5-d58adbb1c047'

cfg.graphs = config_dict.ConfigDict()
cfg.graphs.ref_vals = config_dict.ConfigDict()
cfg.graphs.ref_vals.tab_one = config_dict.ConfigDict()
cfg.graphs.ref_vals.tab_one.items = 223900, 223901, 220739
cfg.graphs.ref_vals.tab_one.label = 'Glascow Coma Score'
cfg.graphs.ref_vals.tab_two = config_dict.ConfigDict()
cfg.graphs.ref_vals.tab_two.items = 223834, 223835, 220339, 224700
cfg.graphs.ref_vals.tab_two.label = 'Ventilation'
cfg.graphs.ref_vals.tab_three = config_dict.ConfigDict()
cfg.graphs.ref_vals.tab_three.items = 220045, 220181, 220210, 220277, 223761
cfg.graphs.ref_vals.tab_three.label = 'Vital Signs'
cfg.graphs.kwargs = config_dict.ConfigDict()
cfg.graphs.kwargs.title_font = 'verdana'
cfg.graphs.kwargs.title_color = 'Black'
cfg.graphs.kwargs.title_size = 25
cfg.graphs.kwargs.text_font = 'verdana'
cfg.graphs.kwargs.text_color = 'Black'
cfg.graphs.kwargs.text_size = 12
cfg.graphs.kwargs.opacity = 0.3
cfg.graphs.kwargs.height = 360
cfg.graphs.kwargs.spikes = True

########################################

cfg.temp = config_dict.ConfigDict()
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


with open('../config-demo.yaml', 'w') as yaml_file:
    yaml.dump(cfg, yaml_file)

print(cfg)
