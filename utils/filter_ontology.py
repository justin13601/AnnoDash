# script to generate filtered ontology csv from loinc.csv or snomed.txt
# result is csv in same format (ie. same columns as source) but just less rows
import os
import numpy as np
import pandas as pd
import pymedtermino
from pymedtermino.snomedct import *

# Ontology (str): "loinc" or "snomed"
# Please note PyMedTermino is required for SNOMED-CT ontology filtering. Please follow installation instructions:
# https://pythonhosted.org/PyMedTermino/tuto_en.html#installation
ontology = "loinc"

# For LOINC, enter classtype (int) ex: 1
# For SNOMED, enter hierarchy level (int) ex: 363787002 (Observable Entity) or 15220000 (Laboratory Test)
ontology_sub = 363787002

if ontology == "loinc":
    print(f"Filtering LOINC codes for classtype: {ontology_sub}.")
elif ontology == "snomed":
    print(f"Filtering SNOMED-CT codes at level: {ontology_sub}.")
    pymedtermino.LANGUAGE = "en"
    pymedtermino.REMOVE_SUPPRESSED_CONCEPTS = True

    df_snomed_new.to_csv('../SNOMED_CT_Observable_Entity.csv', index=False)
