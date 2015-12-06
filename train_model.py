import numpy
import pandas
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score

vectors = pandas.read_pickle('vectors.pkl').values
numpy.random.shuffle(vectors)

train_vectors = vectors[:len(vectors) * 9 // 10, :]
test_vectors = vectors[len(vectors) * 9 // 10:, :]

model = LinearSVC()
model.fit(train_vectors[:, 1:], train_vectors[:, 0])

predicted = model.predict(test_vectors[:, 1:])
score = average_precision_score(test_vectors[:, 0], predicted)
print(score)
