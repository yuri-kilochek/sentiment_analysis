import pandas
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

corpus = pandas.read_pickle('corpus.pkl')
tagged_documents = [TaggedDocument(data['tokens'], [id]) for id, data in corpus.iterrows()]

print('training...')
model = Doc2Vec(tagged_documents,
    workers=8,
)
print('done!')

abstract_feature_names = list(map('abstract_feature_{}'.format, range(model.vector_size)))
vectors = pandas.DataFrame(
    columns=['is_negative'] + abstract_feature_names,
)
vectors['is_negative'] = corpus['is_negative']
vectors = vectors.astype(np.float32)
for i, vector in enumerate(model.docvecs):
    id = model.docvecs.index2doctag[i]
    vectors.loc[id][abstract_feature_names] = vector

vectors.to_pickle('vectors.pkl')

