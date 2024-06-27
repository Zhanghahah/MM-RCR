<div align="center">

<img src="../assets/mm-rcr.pdf" width="500"/>

</div>

# 🐕 SMILES/Graph/Corpus integrated finetuning
MM-RCR model is very similar to standard LLM-based multimodal model. The differences are summarized as follows:
* It requires an extra smiles encoder
* It requires high quality text-image data pairs
* It needs a linear projection layer that connects the visual encoder and LLM


## 🏃 How to train the model
Remember to prepare you data first based on [tutorial](../README.md). 
```bash
 training_scripts/run_7b.sh
 ```



## 👀 A few examples

