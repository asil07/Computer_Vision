from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,
                help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 75)

print("INFO: tuning hyperparameters...")
# defines the set of parameters that I want to tune then start a
# grid search where it evaluate our model for each value of C
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args["jobs"])

model.fit(db["features"][:i], db["labels"][:i])
print(f'Best hyperparameters : {model.best_params_}')

print("INFO: Evaluatin .. .")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds, target_names=[db["label_names"]]))

acc = accuracy_score(db["labels"][i:], preds)
print(f'INFO score: {acc}')

f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))

db.close()

