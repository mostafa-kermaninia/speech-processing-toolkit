from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

def get_knn_model(n_neighbors=2):
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def get_svm_model(kernel='rbf', probability=True, random_state=42):
    return SVC(kernel=kernel, probability=probability, random_state=random_state)

def get_xgboost_model(eval_metric='logloss', random_state=42):
    return XGBClassifier(eval_metric=eval_metric, random_state=random_state)

def get_adaboost_model(n_estimators=50, random_state=42):
    return AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
