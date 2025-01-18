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

# check availility of CJK fonts
# fc-list | grep "Noto Sans CJK"  # check available CJK fonts

cd src
streamlit run app.py
```


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
