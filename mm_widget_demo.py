import sys

import numpy as np

import astropy.modeling.functional_models as models

from sp_model_manager import SpectralModelManagerApp


def debug_print(manager):
    # when GUI is killed, continue execution - here, we print components
    for component in manager.components:
        print component

    # compute spectrum stored in manager
    wave = np.arange(1,10,.1)
    flux = manager.spectrum(wave)
    print flux


def test1():
    # optional initial model is a list of component instances.
    # Note that a tie refers to other components by the 'name' attribute.
    components = [models.Gaussian1D(2.0, 2.0, 2.0, name='test_name_1'), \
                  models.Lorentz1D(3.0, 3.0, 3.0, tied = {'fwhm':lambda m: 1.2 * m['test_name_1'].stddev}),
                  models.GaussianAbsorption1D(0.1, 0.1, 0.1)]

    # start manager and interact with the GUI
    manager = SpectralModelManagerApp(components)

    debug_print(manager)
    

def test2():
    # optional initial model is defined in an import file
    # fname = "/Users/busko/Projects/specfit/proto/n5548_models.py"
    fname = sys.argv[2]

    # start manager and interact with the GUI
    manager = SpectralModelManagerApp(fname)

    debug_print(manager)


def test3():
    manager = SpectralModelManagerApp()

    debug_print(manager)


if __name__ == "__main__":
    exec sys.argv[1]


