import sys

import numpy as np
import astropy.modeling.functional_models as models

from sp_model_manager import SpectralModelManagerApp

# Note all that is being tested is the App version of the model manager.
#
# Successful test sequence:
#  1 - start app with constructor call that reads model from file,
#      either by importing or by passing file name to constructor:
#        - test2 accepts file name
#        - test4 accepts compound model instance imported from file.
#  2 - quit window and check spectrum that is spit by manager.
#  3 - repeat step 1
#  4 - modify parameter value and check spectrum.
#
# Failed test sequence:
#  1 - repeat step 1
#  2 - repeat step 4
#  3 - delete model components
#  4 - read (from Read button) same file as in step 1
#  5 - check that the modified value in step 4 got reset
#  6 - quit window: spectrum is computed from previous parameter value
#  Diagnostic: this is caused by reloading the module and re-generating
#  the GUI from it, but failing to re-define the compound model instance
#  inside the app.
#
#  First we should perhaps modify the basic premise that models read from
#  file should be added to the already existing model. Before that, we
#  should investigate if astropy can handle composite compound models,
#  that is, compound models that can have compound models as components.
#  Maybe that can be captured with parenthesis notation? In any case, how
#  we combine the model in a file with the existing model? Using addition
#  of parenthesised expressions, or just plain addition of bare expressions??


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

    # wave = np.arange(1000.,1400.,5.)
    wave = np.arange(1,10,.1)
    debug_print(manager, wave)


def test3():
    manager = SpectralModelManagerApp()

    wave = np.arange(1,10,.1)
    debug_print(manager, wave)


def test4():
    # import n5548_models
    # compound_model = n5548_models.model1
    import atest1
    compound_model = atest1.model1
    # import test2
    # compound_model = test2.model1

    manager = SpectralModelManagerApp(compound_model)

    # wave = np.arange(1000.,1400.,5.)
    wave = np.arange(1,10,.1)
    debug_print(manager, wave)


if __name__ == "__main__":
    exec sys.argv[1]


