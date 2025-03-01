"""
conda activate zinets

pip install sentence-transformers

cd ~/projects/digital-duck/st_semantics/devs/qwen
python labse.py

"""

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('LaBSE')  # Replace with your chosen model

chinese_sentence = "你好，很高兴认识你。"  # Chinese
chinese_sentence_2 = "很高兴见到你"  # Chinese
english_sentence = "Hello, nice to meet you."  # English
spanish_sentence = "Hola, encantado de conocerte."  # Spanish
german_sentence_1 = "schön dich zu sehen"

sentences = [
    chinese_sentence, english_sentence, spanish_sentence
    , chinese_sentence_2, german_sentence_1
]
embeddings = model.encode(sentences)
similarity = util.cos_sim(embeddings[0], embeddings)
# print("\n".join(sentences))
# print((similarity[0]).tolist())

for x,y in zip(sentences, (similarity[0]).tolist()):
    print(f"{x} \t sim={y}")