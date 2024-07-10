
<iframe src="../assets/mm-rcr.pdf" width="100%" height="500" frameborder="0"></iframe>

# 🐕 SMILES/Graph/Corpus integrated finetuning
MM-RCR model is very similar to standard LLM-based multimodal model. The differences are summarized as follows:
* It requires an extra graph encoder and an smiles encoder
* It requires high quality corpus-smils data pairs
* It needs two linear projection layer (smiles to corpus and graph to corpus)that connects the encoders and LLM
Note that, we commit that we will release the complete  code script if the paper is accepted. Thanks a lot!


## 🏃 How to train the model
Remember to prepare you data first based on [tutorial](../README.md). 
```bash
 training_scripts/run_7b.sh
 ```



## 👀 A few examples

