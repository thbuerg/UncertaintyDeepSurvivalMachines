# riskiano
Risk modelling for CVD via NNs and latent risk factors from VAEs. Powered by UKBB.

# Filestructure
```
├── data
│   ├── ukb_downloads
│   │   ├── master.csv
│   │   ├── ukb_data 
│   │   │   ├── decoded.csv
│   │   │   ├── decoded.bulk
│   │   │   ├── decoded.html
│   │   │   ├── decoded.log
│   │   │   ├── decoded.r
│   │   │   ├── encoding.ukb
│   │   │   └── fields.ukb
│   │   ├── ukb_genetic
│   │   └── ukb_utils
│   ├── ukb_preprocessed
│   │   ├── data_per_table
│   │   │   ├── meta
│   │   │   │   └──  meta.csv 
│   │   │   └── table1
│   │   │       └──  table1.csv 
│   │   └── cleaned??
│   └── datasets
│       ├── baseline_risk
│       │   ├── train.csv
│       │   ├── test.csv
│       │   └── valid.csv
│       └── other...
├── results
├── code                                            # should always be master
│   ├── riskiano                                    # git repo
│   │   ├── README.md                               # this file
│   │   ├── setup.snk                               # workflow that creates everything
│   │   ├── run.sh                                  # calls workflow
│   │   ├── cvd.yaml                                # conda env specifications
│   │   ├── code/                                   # scripts that take input and ouput path args                              
│   ├── funpack                                     # gitlab repo
│   │   ...
...
```

# Setup and Data access
This project is based on data from the UK BioBank cohort. Cohort Baskets need to be downloaded and processed individually.

For processing, `riskiano` relies on `funpack`, a lightweight UKBB parser with simple preprocessing tools.
Once the `decoded.csv` is derived, `setup.snk` applies `funpack` to split the data into data specific tables and take care of simplistic preprocessing.
A requirement for that is a `master.csv` file, defining the data tables (e.g. Blood Assay Results vs ECG measures), NAValues, Parent variables and encodings.

### Codings_Showcase_map.txt:
The `Codings_Showcase_map.txt` is derived by manual annotation (*__sorry__*), and maps the values of individual codings to processing rules such as recoding (e.g. to a different scale) or NA replacement.

### fields.ukb
A file listing all field IDs (one per line), in the current project.

### DataDictionary_Showcase.csv
Is a native UKBB file, that ist downloadable [here](https://biobank.ctsu.ox.ac.uk/~bbdatan/Data_Dictionary_Showcase.csv). It holds information on the individual fields of the database, including variable to coding maps.

### EssentialFields.txt
This is a list of essential fields (one field per line). The **intersection** of these fields determines the subject IDs to be included in the resulting dataset.  
Use this field to filter for instance all IDs that have EKGs or BloodAssays or both (by listing both fields).

### Category files:
The category files (e.g. those in `resources/ukbb_core_categories`) specify which field belongs to which category.  
**Important Note**: The data will be split by category based on these files.

`workflow.snk` executes the snakemake workflow to set up the project.

# Logbook
[here](https://github.com/thbuerg/notes/blob/master/UKBB_UCLEB.md)


