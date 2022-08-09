import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import random

random.seed(42)

df = pd.read_csv('data_mooid.csv', delimiter=';')

df = df.sample(frac=1).reset_index(drop=True)
df = df.replace({',': '.'}, regex=True)

X = df.drop('Hasil', axis=1)
y = df['Hasil']

X = np.array(X, dtype=np.float32)
y = np.array(y)

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)