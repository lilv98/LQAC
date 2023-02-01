# Beyond Knowledge Graphs: Neural Logical Reasoning with Ontologies
Implementation of TAR: A Neural Logical **R**easoner over **T**box and **A**box

# Requirements
* python == 3.8.5
* torch == 1.8.1
* numpy == 1.19.2
* pandas == 1.0.1
* tqdm == 4.61.0
* groovy == 4.0.0
* JVM == 1.8.0_333

# Datasets

## Yago4
### Use pre-processed datasets
Download and unzip YAGO4.zip from [here](https://drive.google.com/drive/folders/1g3_7v-Alzh5o6_3iowt9Auq_3Z916xjL?usp=share_link), and replace

    ./data/YAGO4/input/

### Build from source
Download the raw data:

*T*: https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-class.nt.gz

*A<sub>ee*: https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-facts.nt.gz

*A<sub>ec*: https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-full-types.nt.gz and https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-simple-types.nt.gz

Unzip the raw data to:

    ./data/YAGO4/raw/

Run all cells in:

    ./code/ppc_YAGO4/raw2mid.ipynb
    ./code/ppc_YAGO4/ppc.ipynb


## DBpedia
### Use pre-processed datasets
Download and unzip DBpedia.zip from [here](https://drive.google.com/drive/folders/1g3_7v-Alzh5o6_3iowt9Auq_3Z916xjL?usp=share_link), and replace

    ./data/DBpedia/input/

### Build from source
Download the raw data to:

*T*: http://downloads.dbpedia.org/2016-10/dbpedia_2016-10.nt

*A<sub>ee*: http://downloads.dbpedia.org/2016-10/core-i18n/en/mappingbased_objects_wkd_uris_en.ttl.bz2

*A<sub>ec*: http://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_transitive_wkd_uris_en.ttl.bz2

Unzip the raw data to:

    ./data/DBpedia/raw/

Run all cells in:

    ./code/ppc_DBpedia/raw2mid.ipynb
    ./code/ppc_DBpedia/ppc.ipynb


## Gene Ontology (GO)
### Use pre-processed datasets
Download and unzip GO.zip from [here](https://drive.google.com/drive/folders/1g3_7v-Alzh5o6_3iowt9Auq_3Z916xjL?usp=share_link), and replace

    ./data/GO/input/

### Build from source

# Run
To reproduce the main results, run the following commands:

    nohup python ./code/ABIN.py --dataset YAGO4 >../log/your_log_file 2>&1 &
    nohup python ./code/ABIN.py --dataset DBpedia >../log/your_log_file 2>&1 &

