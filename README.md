<div id="top"></div>

[![Python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://colab.research.google.com/)

[![DOI](https://zenodo.org/badge/490904949.svg)](https://zenodo.org/badge/latestdoi/490904949)

<!-- PROJECT LOGO -->
<br />
<div align="center">

[//]: # (    <img src="assets/mimic.png" alt="Logo" height="80">)

<h3 align="center">AnnoDash</h3>

  <p align="center">
A Clinical Terminology Annotation Dashboard    <br />
    (Supports LOINC®, SNOMED CT, ICD-10-CM, OMOP v5)
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#demo-data">Demo Data</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
</ol>
</details>



<!-- ABOUT THE PROJECT -->

## About

AnnoDash is a deployable clinical terminology annotation dashboard developed primarily in Python using Plotly Dash. It
allows users to annotate medical concepts on a straightforward interface supported by visualizations of associated
patient data and natural language processing.

The dashboard seeks to provide a flexible and customizable solution for clinical annotation. Recent large language
models (LLMs) are supported to aid the annotation process. Additional extensions, such as machine learning-powered
plugins and search algorithms, can be easily added by technical experts.

A demo with ```chartevents``` & ```d_items``` from the MIMIC-IV v2.2 ```icu``` module is available under releases.

[//]: # (loaded is deployed on Heroku [here]&#40;https://mimic-iv-dash-v2.herokuapp.com/&#41; and on Google App Engine [here]&#40;https://mimic-dash-dot-kind-lab.nn.r.appspot.com/&#41;.)

Previously featured
on [Plotly & Dash 500](https://www.linkedin.com/posts/dave-gibbon-8a6219_python-plotly-dash-activity-6993654939717689344-pYrw)!

#### Overview

![Home](assets/dash.png)
The top left section of the dashboard features a dropdown to keep track of target concepts the user wishes to
annotate. The target vocabulary is also selected in a dropdown in this section. The top right module contains the data
visualization component. The bottom half of the dashboard includes modules dedicated to querying and displaying
candidate ontology codes.

#### Data Visualization

| ![graph1](assets/graph1.png)       | ![graph2](assets/graph2.png)        |
|------------------------------------|-------------------------------------|

The dashboard is supported by visualization of relevant patient data. For any given target concept, patient observations
are queried from the source data. The ```Distribution Overview``` tab contains a distribution summarizing all patient
observations. ```Sample Records``` selects the top 5 patients (as ranked by most observations) and displays their
records over a 96-hour window. Both numerical and text data are supported. The format of the source data is detailed
below in <a href="#usage">Usage</a>.

#### Annotation

The user annotates target concepts by first selecting the to-be annotated item in the first dropdown. The following
dropdown allows users to select the target ontology. Several default vocabularies are available, but users are free to
modify the dashboard for additional ontology support via scripts detailed in <a href="#other-relevant-files">Other
Relevant Files</a>. Code suggestions are then generated in the bottom table. Users are able to select their target
annotation via clicking and the appropriate data is saved in ```.json``` files after submission.

#### Ontology Search & Ranking

The dashboard automatically generates ontology code suggestions based on the target concept. A string search supported
by PyLucene and the Porter stemming algorithm sorts results by relevance, as indicated by the colour of the circle
icons. Several other methods of string search are available, such as full text search using SQLite3's FTS5 or
ElasticSearch, vector search using TF-IDF, and similarity scoring using Jaro-Winkler/Fuzzy partial ratios. NLM UMLS API
is also available for the SNOMED CT ontology.

After searching, the dashboard is able to re-rank ontology codes using LLMs. Currently, OpenAI's GPT-3.5 API and
CohereAI's re-ranking API endpoint is supported by default. LLM re-ranking is disabled by default; however, if desired,
API keys will be required along with associated costs.

<p align="right">(<a href="#top">back to top</a>)</p>
<!-- GETTING STARTED -->

## Getting Started

Below are steps to download, install, and run the dashboard locally. Leave all configuration fields unchanged to run the
demo using MIMIC-IV data.

### Requirements

The dashboard requires the following major Python packages to run:

* [Dash][dash]~=2.6.0
* [Pandas][pandas]~=1.4.2
* [Plotly][plotly]~=5.8.0
* [NumPy][numpy]~=1.22.3
* [PyYAML][pyyaml]~=6.0
* [SciPy][scipy]~=1.7.3

All other packages are listed in ```requirements.txt```.

Additionally, the latest version of the dashboard requires PyLucene for its primary ontology code searching algorithm.
Please follow setup instructions available [here](https://lucene.apache.org/pylucene/install.html).

#### Required Files:

* A ```.csv``` file containing all patient observations/data (missingness allowed, except for the ```itemid``` column):
  ```
  itemid,subject_id,charttime,value,valueuom
  52038,123,2150-01-01 10:00:00,5,mEq/L
  52038,123,2150-01-01 11:00:00,6,ug/mL
  ...
  ```
* A ```.csv``` file containing all concepts to be annotated in id-label pairs, {id: label}:
  ```
  itemid,label
  52038,Base Excess
  52041,pH
  ...
  ```
* The ```config.yaml```:
    * Define results directory (default: ```/results-json/demo```)
    * Define location of the source data ```.csv``` (default: ```/demo-data/CHARTEVENTS.csv```)
    * Define location of the concepts ```.csv``` (default: ```/demo-data/demo_chartevents_user_1.csv```)
    * Define location of ontology SQLite3 databases (default: ```/ontology```)
    * Define string search algorithm (default: ```pylucene```)
    * Define ranking algorithm (default: ```None```)
    * Define dashboard aesthetics for graphs (defaults are shown in the configuration file)

#### Using ElasticSearch:

To utilize ElasticSearch as the string search algorithm, run a local ElasticSearch cluster via Docker and specify '
elastic' in the appropriate configuration field:

   ```sh
   docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0
   ```

#### Using APIs:

If desired, please define your API keys (OpenAI, CohereAI, NLM UMLS) as environment variables prior to running the
dashboard. This can be done explicitly via editing the Docker Compose file below.

<p align="right">(<a href="#top">back to top</a>)</p>

### Installation
#### Docker Install (Recommended):

1. Clone repository:
   ```sh
   git clone https://github.com/justin13601/AnnoDash.git
   ```

2. Edit ```/src/generate_config.py``` with desired directories and configurations and run:
    ```sh
   python3 generate_config.py
    ```
   This creates the ```config.yaml``` required by the dashboard.


3. Build dashboard image:
   ```sh
   docker build -t annodash .
   ```


4. Modify docker-compose.yml to include API keys where necessary. Then, run Docker container:
   ```sh
   docker-compose up
   ```

#### Manual Install:

1. Clone repository:
   ```sh
   git clone https://github.com/justin13601/AnnoDash.git
   ```

2. Install requirements:
   ```sh
   pip install -r requirements.txt
   ```

3. Install PyLucene and associated Java libraries.
    ```sh
    # use shell scripts to install jcc and pylucene
    ```

4. Edit ```/src/generate_config.py``` with desired directories and configurations and run:
    ```sh
   python3 generate_config.py
    ```
   This creates the ```config.yaml``` required by the dashboard.

5. Run dashboard:
   ```sh
   python3 main.py
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage

Install/run the dashboard and visit http://127.0.0.1:8080/ or http://localhost:8080/.

#### Other Relevant Files

```/src/generate_config.py``` is used to generate the ```config.yaml``` file.

```/src/generate_ontology_database.py``` uses SQLite3 to generate the ```.db``` database files used to store the
ontology vocabulary. This is needed when defining custom vocabularies outside the default list of available ones.

```/src/generate_pylucene_index.py``` is used to generate the index used by PyLucene for ontology querying. This is
needed when defining custom vocabularies outside the default list of available ones.

```/src/generate_elastic_index.py``` is used to generate the index used by ElasticSearch for ontology querying. This is
needed when defining custom vocabularies outside the default list of available ones. This can be run only after a local
ElasticSearch cluster is created via Docker.

```/src/search.py``` includes classes for ontology searching.

```/src/rank.py``` includes classes for ontology ranking.

<p align="right">(<a href="#top">back to top</a>)</p>

## Demo Data

Demo data and respective licenses are included in the [demo-data folder](/demo-data).

- MIMIC-IV Clinical Database demo is available on Physionet (Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi,
  L. A., & Mark, R. (2023). MIMIC-IV Clinical Database Demo (version 2.2).
  PhysioNet. https://doi.org/10.13026/dp1f-ex47).

- LOINC® Ontology Codes are available at https://loinc.org.

- SNOMED CT Ontology Codes are available at https://www.nlm.nih.gov/healthit/snomedct/index.html.

- ICD-10-CM Codes are available at https://www.cms.gov/medicare/icd-10/2022-icd-10-cm.

- OMOP v5 Codes are available at https://athena.ohdsi.org/search-terms/start.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the [MIT][mit] License.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

* Alistair Johnson, DPhil | The Hospital for Sick Children | Scientist
* Mjaye Mazwi, MBChB, MD | The Hospital for Sick Children | Staff Physician
* Danny Eytan, MD, PhD | The Hospital for Sick Children | Staff Physician
* Oshri Zaulan, MD | The Hospital for Sick Children | Staff Intensivist
* Azadeh Assadi, MN | The Hospital for Sick Children | Pediatric Nurse Practitioner

<p align="right">(<a href="#top">back to top</a>)</p>


[dash]: https://dash.plotly.com/installation

[pandas]: https://pandas.pydata.org/docs/getting_started/install.html

[plotly]: https://plotly.com/python/getting-started/

[numpy]: https://numpy.org/install/

[pyyaml]: https://pyyaml.org/wiki/PyYAMLDocumentation

[scipy]: https://scipy.org/install/

[mit]: https://opensource.org/licenses/MIT
