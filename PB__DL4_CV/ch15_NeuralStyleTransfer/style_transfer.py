from pyimagesearch.nn.conv.neuralstyle import NeuralStyle
from tensorflow.keras.applications import VGG19

SETTINGS = {
    "input_path": "input/face_noun.jpg",
    "style_path": "input/mcescher.jpg",
    "output_path": "output",

    # CNN to be used for style transfer
    "net": VGG19,
    "content_layer": "block4_conv2",
    "style_layers": ["block1_conv1", "block2_conv1", "block3_conv1",
                     "block4_conv1", "block5_conv1"],
    "content_weight": 1.0,
    "style_weight": 100.0,
    "tv_weight": 10.0,

    "iterations": 50
}

ns = NeuralStyle(SETTINGS)
ns.transfer()
