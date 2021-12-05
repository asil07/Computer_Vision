import matplotlib
matplotlib.use("Agg")


from conf import sr_config as config
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.srcnn import SRCNN
from  tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


def super_res_generator(inputDataGen, targetDataGen):
    while True:

        inputData = next(inputDataGen)[0]
        targetData = next(targetDataGen)[0]

        yield (inputData, targetData)


inputs = HDF5DatasetGenerator(config.INPUT_DB, config.BATCH_SIZE)
targets = HDF5DatasetGenerator(config.OUTPUT_DB, config.BATCH_SIZE)

print("INFO: compiling model ...")

opt = Adam(learning_rate=0.001, decay=0.001/config.NUM_EPOCHS)
model = SRCNN.buil(width=config.INPUT_DIM, height=config.INPUT_DIM, depth=3)
model.compile(loss="mse", optimizer=opt)

H = model.fit_generator(
    super_res_generator(inputs.generator(), targets.generator()),
    steps_per_epoch=inputs.numImages // config.BATCH_SIZE, epochs=config.NUM_EPOCHS, verbose=1

)

print("INFO: serializing model... ")

model.save(config.MODEL_PATH, overwrite=True)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["loss"], label="loss")
plt.title("Loss on super resolution traiing")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(config.PLOT_PATH)

inputs.close()
targets.close()



