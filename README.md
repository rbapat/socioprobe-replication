# socioprobe-replication
Final Project for COMPSCI 685: Advanced NLP, Spring 2023  
Group: Rohan Bapat, Fareya Ikram, Linus Jen & Jocelyn Lutes


* This project is centered around replicating and expanding upon some of the experiments done in [SocioProbe](https://aclanthology.org/2022.emnlp-main.539/) by Lauscher et al.

* Specifically, the main goals of this project were to explore if sociodemographic factors, such as `age` and `gender` are encoded by large lanugage models and (if possible) in which layers most of the knowledge is located. 
* This repository provides code to probe embeddings created with  `roberta-base`, `microsoft/electra-base-discriminator`, and `google/deberta-base` for sociodemographic knowledge. 

### Overview of Process
1. Obtain pre-trained models from `HuggingFace`
2. Use pre-trained models to create contextual embeddings of the input text
2. Feed the frozen embeddings to a probing model
3. Evaluate Test F1 score and Test MDL score for each dataset
4. Plot the results 

### Getting Started

1. Install the necessary packages using:
```
pip install -r requirements.txt
```
2. Move necessary data to the `data/` directory

2. Decide which dataset you would like to probe and run:
```
python3 general_linus/main_{dataset}.py
```

The `general_linus/main_{dataset}.py` script contains all of the necessary code to create embeddings, build, train, and evaluate probes, and create plots of the results. 

### Additional Functionality
- This repository also contains a Jupyter Notebook that allows you to make requests to the OpenAI API to determine if GPT-3 and ChatGPT are able to reason about sociodemographic variables, such as `age` and `gender`.