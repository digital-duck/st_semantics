
## Features

1. Using **state-of-the-art multilingual embedding models**.
2. Working with a dataset of **Chinese characters/phrases (≤3 characters)** and their **English translations**.
3. Starting with **non-linear dimensionality reduction** (t-SNE or Isomap) for 2D visualization.
4. Skipping clustering initially to simplify the study.

Here’s the step-by-step plan:

---

### Step 1: Choose Multilingual Embedding Models
We’ll use **two of the best multilingual embedding models**:
1. **XLM-R (XLM-RoBERTa)**: A powerful transformer-based model trained on 100+ languages.
2. **LASER (Language-Agnostic SEntence Representations)**: Specifically designed for multilingual embeddings and cross-lingual tasks.

---

### Step 2: Prepare the Dataset
- Your dataset consists of:
  - Chinese characters/phrases (≤3 characters).
  - Their English translations.
- Example:
  ```
  Chinese: 你好, 爱, 天气
  English: Hello, Love, Weather
  ```

---

### Step 3: Extract Embeddings
- Use the chosen models to extract embeddings for both the Chinese and English words.
- For XLM-R, we’ll use the Hugging Face `transformers` library.
- For LASER, we’ll use the `laserembeddings` Python package.

---

### Step 4: Apply Non-Linear Dimensionality Reduction
- Use **t-SNE** or **Isomap** to reduce the embeddings to 2D for visualization.
- t-SNE is great for local structure preservation, while Isomap is better for global structure.

---

### Step 5: Visualize the 2D Scatter Plot
- Plot the reduced embeddings to see how Chinese characters/phrases and their English translations are distributed in the embedding space.

---

### Implementation in Python

#### Install Required Libraries
```bash
pip install transformers laserembeddings numpy matplotlib scikit-learn umap-learn
```

#### Code Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from laserembeddings import Laser
from sklearn.manifold import TSNE, Isomap

# Step 1: Prepare the dataset
chinese_words = ["你好", "爱", "天气", "书", "猫"]  # Example Chinese words/phrases
english_words = ["Hello", "Love", "Weather", "Book", "Cat"]  # Corresponding English translations

# Step 2: Extract embeddings using XLM-R
def get_xlmr_embeddings(words, model_name="xlm-roberta-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []
    for word in words:
        inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return np.vstack(embeddings)

# Extract embeddings for Chinese and English words
chinese_embeddings_xlmr = get_xlmr_embeddings(chinese_words)
english_embeddings_xlmr = get_xlmr_embeddings(english_words)

# Step 3: Extract embeddings using LASER
laser = Laser()
chinese_embeddings_laser = laser.embed_sentences(chinese_words, lang="zh")
english_embeddings_laser = laser.embed_sentences(english_words, lang="en")

# Step 4: Combine embeddings for visualization
embeddings_xlmr = np.vstack([chinese_embeddings_xlmr, english_embeddings_xlmr])
embeddings_laser = np.vstack([chinese_embeddings_laser, english_embeddings_laser])

# Step 5: Apply t-SNE for dimensionality reduction
def reduce_dimensions(embeddings, method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "isomap":
        reducer = Isomap(n_components=2)
    else:
        raise ValueError("Unsupported method. Use 'tsne' or 'isomap'.")
    return reducer.fit_transform(embeddings)

# Reduce dimensions for XLM-R and LASER embeddings
reduced_xlmr = reduce_dimensions(embeddings_xlmr, method="tsne")
reduced_laser = reduce_dimensions(embeddings_laser, method="tsne")

# Step 6: Visualize the 2D scatter plot
def plot_embeddings(reduced_embeddings, labels, title):
    plt.figure(figsize=(10, 6))
    for i, (x, y) in enumerate(reduced_embeddings):
        plt.scatter(x, y, label=labels[i])
    plt.title(title)
    plt.legend()
    plt.show()

# Labels for the plot
labels = chinese_words + english_words

# Plot XLM-R embeddings
plot_embeddings(reduced_xlmr, labels, "XLM-R Embeddings (t-SNE)")

# Plot LASER embeddings
plot_embeddings(reduced_laser, labels, "LASER Embeddings (t-SNE)")
```

---

### Explanation of the Code
1. **Embedding Extraction**:
   - For XLM-R, we average the token embeddings to get a single vector per word.
   - For LASER, we use the `laserembeddings` library to directly get sentence-level embeddings.

2. **Dimensionality Reduction**:
   - We use t-SNE by default, but you can switch to Isomap by changing the `method` parameter in `reduce_dimensions`.

3. **Visualization**:
   - The 2D scatter plot shows how Chinese and English words are distributed in the embedding space.

---

### Next Steps
1. Analyze the scatter plots to see if:
   - Chinese words and their English translations are close in the embedding space.
   - There are any interesting patterns or outliers.
2. Experiment with **Isomap** or **UMAP** for dimensionality reduction.
3. Extend to **3D visualization** or **automatic clustering** once you’re comfortable with the initial results.

