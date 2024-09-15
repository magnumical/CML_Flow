import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=2,n_redundant=8, n_classes=2, random_state=42)

np.unique(y, return_counts=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params= {
    "solver": "lbfgs", "max_iter": 1000, "random_state": 42, "multi_class": "auto",
}


lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

