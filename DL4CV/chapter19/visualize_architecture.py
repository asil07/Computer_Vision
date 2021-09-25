from pyimagesearch.nn.conv.lenet import LeNet
from tensorflow.keras.utils import plot_model
import pydot_ng as pydot
pydot.find_graphviz()


model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenet.png", show_shapes=True)
