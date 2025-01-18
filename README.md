# st_semantics

Study Semantics of Natural Languages, especially, Chinese and English.


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
cd src
python -m laserembeddings download-models
pip install laserembeddings[zh]

fc-list | grep "Noto Sans CJK"  # check available CJK fonts

streamlit run app.py
```

