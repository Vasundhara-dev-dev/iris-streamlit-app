from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

iris = load_iris()
x, y = iris.data, iris.target

model = LogisticRegression(max_iter=200)
model.fit(x, y)

joblib.dump(model, 'iris_model.pkl')
