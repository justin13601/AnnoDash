# MIMIC-IV Clinical Dashboard

Clinical dashboard created using Plotly Dash &amp; MIMIC-IV database.

[![DOI](https://zenodo.org/badge/490904949.svg)](https://zenodo.org/badge/latestdoi/490904949)

Table of Contents
-----------------

* [Requirements](#requirements)
* [Usage](#usage)
* [Demo Data](#demo-data)
* [License](#license)

Requirements
------------

The dashboard requires the following to run:

* [Dash][dash]~=2.4.1
* [Pandas][pandas]~=1.4.2
* [Plotly][plotly]~=5.8.0
* [NumPy][numpy]~=1.22.3
* [PyYAML][pyyaml]~=6.0
* [SciPy][scipy]~=1.7.3

Usage
------------

```
python3 app.py
```

Files required:

* A .csv file containing all patient observations
* A .csv file containing all lab measurements to-be annotated in id-label pairs {id: label}
* config.yaml:
    * Define results directory
    * Define data directory
    * Define ontology directory (location of LOINC.csv etc.)
    * Define up to 3 pairs of lab measurements to plot annotations against (defaults are indicated)

| ![Home](assets/home.png)          | ![Tabs](assets/tabs.png)          |
|-----------------------------------|-----------------------------------|
| ![Annotate](assets/annotate1.png) | ![Annotate](assets/annotate2.png) |

Demo Data
------------
Demo data and respective licenses are included in the [demo-data folder](/demo-data).

- MIMIC-IV Clinical Database demo is available on Physionet (Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2022). MIMIC-IV Clinical Database Demo (version 1.0). PhysioNet. https://doi.org/10.13026/jwtp-v091).

- LOINC Ontology Codes are available at https://loinc.org.


Licenses
------------
Licensed under the [MIT][mit] license.



[dash]: https://dash.plotly.com/installation

[pandas]: https://pandas.pydata.org/docs/getting_started/install.html

[plotly]: https://plotly.com/python/getting-started/

[numpy]: https://numpy.org/install/

[pyyaml]: https://pyyaml.org/wiki/PyYAMLDocumentation

[scipy]: https://scipy.org/install/

[mit]: https://opensource.org/licenses/MIT

