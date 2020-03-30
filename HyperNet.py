# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data: 
import tensorflow as tf

def getWeightVariable(shape, name):
    return tf.get_variable(shape=shape, name=name)

def getBiasVariable(shape, name):
    return tf.get_variable(shape=shape, name=name)


def genWeight(layerSize, input, weightSize):
    layers = [input, ]
    for i in range(layerSize+1):
        w = getWeightVariable(weightSize[i], 'layer{:d}_weights'.format(i+1),)
        b = getBiasVariable(weightSize[i][1], 'layer{:d}_bias'.format(i+1),)
        layers.append(tf.add(tf.tensordot(layers[i], w, [[-1], [-2]]), b))
    return layers[-1]
