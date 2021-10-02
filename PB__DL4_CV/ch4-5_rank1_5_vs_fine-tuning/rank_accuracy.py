from pyimagesearch.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

print("INFO: loading pre-trained network...")
model = pickle.loads(open(args["model"], "rb").read())

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

print("INFO: predicting...")
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(preds, db['labels'][i:])

print(f"INFO: rank-1: {rank1 * 100:.2f}")
print(f"INFO: rank-5: {rank5 * 100:.2f}")

db.close()














