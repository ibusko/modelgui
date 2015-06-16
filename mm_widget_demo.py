import sys

import numpy as np
import astropy.modeling.functional_models as models

from sp_model_manager import SpectralModelManagerApp


def debug_print(manager, wave):
    # when GUI is killed, continue execution - here, we print components
    for component in manager.components:
        print component

    # compute and print spectrum stored in manager
    print(manager.spectrum(wave))


def test1():
    # optional initial model is a list of component instances.
    # Note that a tie refers to other components by the 'name' attribute.
    components = [models.Gaussian1D(1., 5., 1., name='test_name_1'), \
                  models.Lorentz1D(0.5, 6., 0.7, tied = {'fwhm':lambda m: 1.2 * m['test_name_1'].stddev}),
                  models.GaussianAbsorption1D(0.5, 3., 1.)]

    # start manager and interact with the GUI
    manager = SpectralModelManagerApp(components)

    wave = np.arange(1,10,.1)
    debug_print(manager, wave)
    

def test2():
    # optional initial model is defined in an import file
    # fname = "/Users/busko/Projects/specfit/proto/n5548_models.py"
    fname = sys.argv[2]

    # start manager and interact with the GUI
    manager = SpectralModelManagerApp(fname)

    wave = np.arange(1000.,1400.,5.)
    debug_print(manager, wave)


def test3():
    manager = SpectralModelManagerApp()

    wave = np.arange(1,10,.1)
    debug_print(manager, wave)


def test4():
    import n5548_models
    compound_model = n5548_models.model1

    manager = SpectralModelManagerApp(compound_model)

    wave = np.arange(1000.,1400.,5.)
    debug_print(manager, wave)


if __name__ == "__main__":
    exec sys.argv[1]


