import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import warnings
import mlflow
warnings.filterwarnings("ignore")


X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=2,n_redundant=8, n_classes=2, random_state=42)

np.unique(y, return_counts=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params= {
    "solver": "lbfgs", 
    "max_iter": 1000, 
    "random_state": 42, 
    "multi_class": "auto",
}


lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
report_dict = classification_report(y_test, y_pred, output_dict=True)



rf_clf = RandomForestClassifier(random_state=42,n_estimators=30)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred_rf)) 

models = [
    ('lr', lr,(X_train, y_train), (X_test, y_test)), 
    ('rf', rf_clf,(X_train, y_train), (X_test, y_test))
]


reports =[]
for model_name, model, train, test in models:
    model.fit(train[0], train[1])
    y_pred = model.predict(test[0])
    report = classification_report(test[1], y_pred,output_dict=True)
    reports.append(report)
    print(f"Model: {model_name}")
    print(report)
    print("---------------------------------------------------")

mlflow.set_experiment("multiple models")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

for i, element in enumerate(models):
    model_name = element[0]
    model = element[1]
    report = reports [i]

    with mlflow.start_run(run_name = model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", report['accuracy'])
        mlflow.log_metric("recall_class_0", report['0']['recall'])
        mlflow.log_metric("recall_class_1", report['1']['recall'])
        mlflow.log_metric("f1_score_macro", report['macro avg']['f1-score'])
  
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
    