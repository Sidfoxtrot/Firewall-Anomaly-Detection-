import pandas as pd
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import SparseCoder




df = pd.read_csv('sparse.csv')
d=df.as_matrix()

dict=DictionaryLearning(n_components=2, max_iter=100)
dict.fit(d)


sp = SparseCoder(dict.components_)

sp.transform(d)






