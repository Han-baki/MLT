"""Some utils related to Keras models."""

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import h5py
import os


def load_weights(model, filepath, layer_names=None):
    """Loads layer weights from a HDF5 save file.
     
    # Arguments
        model: Keras model
        filepath: Path to HDF5 file
        layer_names: List of strings, names of the layers for which the 
            weights should be loaded. List of tuples 
            (name_in_file, name_in_model), if the names in the file differ 
            from those in model.
    """
    filepath = os.path.expanduser(filepath)
    f = h5py.File(filepath, 'r')
    if layer_names == None:
        layer_names = [s.decode() for s in f.attrs['layer_names']]
    for name in layer_names:
        if type(name) in [tuple, list]:
            layer_name = name[1]
            name = name[0]
        else:
            layer_name = name
        g = f[name]
        weights = [np.array(g[wn]) for wn in g.attrs['weight_names']]
        try:
            layer = model.get_layer(layer_name)
            #assert layer is not None
        except:
            print('layer missing %s' % (layer_name))
            print('    file  %s' % ([w.shape for w in weights]))
            continue
        try:
            #print('load %s' % (layer_name))
            layer.set_weights(weights)
        except Exception as e:
            print('something went wrong %s' % (layer_name))
            print('    model %s' % ([w.shape.as_list() for w in layer.weights]))
            print('    file  %s' % ([w.shape for w in weights]))
            print(e)
    f.close()

