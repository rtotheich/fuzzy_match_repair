from gensim.test.utils import common_texts
from gensim.models import Word2Vec

our_model = Word2Vec(common_texts, vector_size=10, window=5, min_count=1, workers=4)
print(our_model.wv.most_similar('computer', topn=5))
