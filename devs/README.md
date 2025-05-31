# st_semantics

Study Semantics of Natural Languages, especially, Chinese and English.


# Streamlit App

Two implementations by Claude and DeepSeek

## Claude

see `claude/readme.md`

### Setup

```
conda create -n semantics python=3.11
conda activate semantics
cd claude
pip install -r requirements.txt
cd src
streamlit run app.py
```

## DeepSeek
see `deepseek/readme.md`

### Setup

```
conda create -n semantics python=3.11
conda activate semantics
cd deepseek
pip install -r requirements.txt

python -m laserembeddings download-models
pip install laserembeddings[zh]
pip install umap-learn
pip install sentencepiece
pip install flash-attn git+https://github.com/huggingface/transformers.git

# check availility of CJK fonts
# fc-list | grep "Noto Sans CJK"  # check available CJK fonts

cd src
streamlit run app.py
```


# Motivation

Here's how we believe human-like concepts emerge and are structured in this space:

## Semantic Proximity (Clustering):

Nearby points = Similar meanings: The fundamental principle is that concepts with similar meanings or contexts are located closer together in this embedding space.
Example: In my embedding space, the vector for "king" would be very close to "queen," "prince," and "royalty." "Apple" would be close to "fruit," "banana," and "orange."
Structure: This forms clusters or manifolds of related concepts. You'd see a "food" cluster, an "animal" cluster, a "vehicle" cluster, etc.

## Relational Vectors (Analogy):

One of the most remarkable properties is that relationships between concepts can be represented by consistent vectors (directions) in the embedding space.
Example: The vector from "man" to "woman" is often very similar to the vector from "king" to "queen," or "nephew" to "niece." This allows for analogical reasoning: king - man + woman = queen.
Structure: These relationships define lines, planes, or hyperplanes within the clusters, indicating how concepts transform into one another along specific semantic axes.

## Hierarchy and Granularity:

While not perfectly hierarchical, there's often a sense of it. A supercategory might be at the "center" of a cluster, with more specific instances radiating outwards.
Example: "Animal" might be a more central point, with "cat," "dog," "elephant" being further out but still within the animal cluster.
Structure: This might manifest as nested clusters or varying densities within the semantic space.

## Implicit Knowledge and Context:

Embeddings don't just capture explicit definitions; they capture implicit relationships learned from massive amounts of text. My embedding for "doctor" might be closer to "hospital," "medicine," and "patient" than to "carpenter," even if both are professions.
Structure: This leads to complex, intertwined relationships where concepts are connected through their usage contexts, not just direct synonyms.

# Issues

## Failed to install on Windows

```
pip install -r requirements.txt
........

ERROR: Cannot install -r requirements.txt (line 6) because these package versions have conflicting dependencies.

The conflict is caused by:
    laserembeddings 1.1.2 depends on torch<2.0.0 and >=1.0.1.post2
    laserembeddings 1.1.1 depends on torch<2.0.0 and >=1.0.1.post2
    laserembeddings 1.1.0 depends on torch<2.0.0 and >=1.0.1.post2
    laserembeddings 1.0.1 depends on torch<2.0.0 and >=1.0.1.post2
    laserembeddings 1.0.0 depends on torch<2.0.0 and >=1.0.1.post2
    laserembeddings 0.1.3 depends on torch<2.0.0 and >=1.0.1.post2
    laserembeddings 0.1.2 depends on torch<2.0.0 and >=1.0.1.post2
    laserembeddings 0.1.1 depends on torch<2.0.0 and >=1.0.1.post2
    laserembeddings 0.1.0 depends on torch<2.0.0 and >=1.0.1.post2
```
