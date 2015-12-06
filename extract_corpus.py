from pathlib import Path
from itertools import chain, repeat
from pandas import DataFrame
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

corpus = DataFrame(columns=['is_negative', 'tokens'])

for is_negative, review_path in chain(
    zip(repeat(0), Path('aclImdb/train/pos').iterdir()),
    zip(repeat(1), Path('aclImdb/train/neg').iterdir()),
    zip(repeat(0), Path('aclImdb/test/pos').iterdir()),
    zip(repeat(1), Path('aclImdb/test/neg').iterdir()),
    ):
    with review_path.open(encoding='UTF-8') as review_file:
        file = str(review_path.relative_to('aclImdb'))
        tokens = word_tokenize(BeautifulSoup(review_file.read()).text)
        corpus.loc[file] = is_negative, tokens 
        print(len(corpus))

corpus.to_pickle('corpus.pkl')
        
