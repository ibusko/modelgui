from __future__ import division

import os
import math
import re

import numpy as np
from astropy.modeling import Parameter, Fittable1DModel

import signal_slot
import models_registry
import sp_adjust
import sp_model_io

from PyQt4.QtCore import *
from PyQt4.QtGui import *

AVAILABLE_COMPONENTS = "Available components"
FONT_SIZE_INCREASE = 2

# To memorize last visited directory.
_model_directory = os.environ["HOME"]


# Builds a compound model by adding together all components in
# the list. This is used when the input offers no clue on how
# to combine the components.

def _buildSummedCompoundModel(components):
    if len(components) < 1:
        return None
    result = components[0]
    if len(components) > 1:
        for component in components[1:]:
            result += component
    return result


# Finds the level at which a tree is being selected.
# Also finds the index for the zero-th level parent
# that is associated with the selected item.

def findLevelAndIndex(indices):
    if len(indices) <= 0:
        return -1, -1
    level = 0
    index0 = -1
    if len(indices) > 0:
        index0 = indices[0]
        while index0.parent().isValid():
            index0 = index0.parent()
            level += 1
    return level, index0



# Subclasses QTreeView in order to detect selections on tree elements
# and enable/disable other GUI elements accordingly.

class _MyQTreeView(QTreeView):
    def __init__(self, model):
        super(_MyQTreeView, self).__init__()
        # Model is required so we can have access to all
        # rows, not just the currently selected one.
        self.model = model

        # Default behavior is the same as the superclass'.
        self.b_up = None
        self.b_down = None
        self.b_delete = None

    # By providing button instances, the specialized
    # behavior is triggered on.
    def setButtons(self, b_up, b_down, b_delete, b_save, model):
        self.b_up = b_up
        self.b_down = b_down
        self.b_delete = b_delete
        self.b_save = b_save
        self.b_up.setEnabled(False)
        self.b_down.setEnabled(False)
        self.b_delete.setEnabled(False)
        self.b_save.setEnabled(model and len(model.items) > 0)

    # Overrides QTreeView to provide
    # sensitivity to selection events.
    def selectionChanged(self, selected, deselected):
        # IndexError may happen in normal GUI usage and it's normal.
        try:
            self._handleTreeSelectionEvent(selected, deselected)
        except IndexError:
            pass

    # Overrides QTreeView to capture and handle a data changed event.
    # These data changes occur in the model associated with the tree,
    # when a Data object gets changed, such as when the user types in
    # a new value for a Parameter instance.
    def dataChanged(self, top, bottom):
        self.emit(SIGNAL("dataChanged"), 0)
        super(_MyQTreeView, self).dataChanged(top, bottom)

    # Here is the logic to gray out buttons based on context.
    def _handleTreeSelectionEvent(self, selected, deselected):

        # only acts if actual button instances exist.
        if self.b_up and self.b_down and self.b_delete:

            # nothing is selected, so no button action is allowed.
            if selected.count() == 0 or selected.count() > 1:
                self.b_up.setEnabled(False)
                self.b_down.setEnabled(False)
                self.b_delete.setEnabled(False)

            # one row is selected, but behavior depends
            # on how many rows there are in the tree.
            else:
                # if a row is selected, it can always be deleted.
                self.b_delete.setEnabled(True)

                if self.model.rowCount() == 1:
                    # only one row in tree, thus only deletion is allowed.
                    self.b_up.setEnabled(False)
                    self.b_down.setEnabled(False)
                else:
                    # two or more rows in tree; see which one is selected.
                    # Watch out for the level though, must be root level.
                    level, index0 = findLevelAndIndex(selected.indexes())
                    row = index0.row()
                    if row > 0 and row < (self.model.rowCount() - 1):
                        # selected row is in the middle; can be moved up or down.
                        self.b_up.setEnabled(True)
                        self.b_down.setEnabled(True)
                    elif row == 0:
                        # selected row is top row; can only be moved down.
                        self.b_up.setEnabled(False)
                        self.b_down.setEnabled(True)
                    else:
                        # selected row is bottom row; can only be moved up.
                        self.b_up.setEnabled(True)
                        self.b_down.setEnabled(False)


# Base window that holds a tree widget that supports contextual menus.
# It needs an instance of QStandardItemModel in order to build the tree.

class _BaseWindow(QWidget):
    def __init__(self, model):
        QWidget.__init__(self)

        self.model = model

        font = QFont(self.font())
        font.setPointSize(font.pointSize() + FONT_SIZE_INCREASE)
        self.setFont(font)
        QToolTip.setFont(font)

        self.treeView = _MyQTreeView(self.model)
        self.treeView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeView.customContextMenuRequested.connect(self.openMenu)
        self.treeView.setModel(self.model)

        grid_layout = QGridLayout()
        grid_layout.addWidget(self.treeView, 0, 0)

        # the following are not used by this class but provide
        # places where sub classes can put in their own widgets.
        self.expression_layout = QHBoxLayout()
        grid_layout.addLayout(self.expression_layout, 1, 0)

        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch()
        grid_layout.addLayout(self.button_layout, 2, 0)

        self.setLayout(grid_layout)

    def openMenu(self, position):
        raise NotImplementedError

    # Returns the selected model.
    def getSelectedModel(self):
        indexes = self.treeView.selectedIndexes()
        if len(indexes) > 0:
            level, index = self.findTreeLevel()
            if len(indexes) > 0:
                data = self.model.item(index.row()).item
                return data
        return None

    # Returns the level at which the tree is being selected.
    # Also returns the index for the zero-th level parent that
    # is associated with the selected item.
    def findTreeLevel(self):
        indices = self.treeView.selectedIndexes()
        return findLevelAndIndex(indices)

    # Connects a slot to the "triggered" signal in a QWidget.
    # This is used to associate callbacks to contextual menus.
    def createAction(self, widget, text, slot=None, shortcut=None,
                     icon=None, tip=None, checkable=False):
        action = QAction(text, widget)
        action.setCheckable(checkable)
        if icon is not None:
            action.setIcon(QIcon("/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        return action


# Window with the active spectral components -------------------------------------------

class _SpectralModelsGUI(object):
    def __init__(self, components):
        self.model = ActiveComponentsModel(components, name="Active components")
        self.window = _SpectralModelsWindow(self.model)

        self.mapper = QDataWidgetMapper()
        self.mapper.setModel(self.model)
        self.mapper.addMapping(self.window.treeView, 0)

        # TODO use QDataWidgetMapper
        # this violation of MVC design principles is necessary
        # so our model manager class can work with the modified
        # code in modelmvc.py. This probably could be done via
        # the QDataWidgetMapper stuff instead.
        self.model.setWindow(self.window)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def updateModel(self, component):
        self._model.addOneElement(component)

        # new components are added to existing compound model
        if hasattr(self._model, 'compound_model'):
            self._model.compound_model = self._model.compound_model + component
        else:
            self._model.compound_model = component
        self.window.updateExpressionField(self._model.compound_model )

        self.window.emit(SIGNAL("treeChanged"), 0)

    def getSelectedModel(self):
        return self.window.getSelectedModel()


class _SpectralModelsWindow(_BaseWindow):
    def __init__(self, model):
        super(_SpectralModelsWindow, self).__init__(model)

        # Contextual menus do not always work under ipython
        # non-block mode. These buttons are an alternative
        # way of implementing the same actions.

        up_button = QPushButton('Up', self)
        up_button.setFocusPolicy(Qt.NoFocus)
        up_button.setToolTip('Moves selected component up')
        self.connect(up_button, SIGNAL('clicked()'), self.moveComponentUp)
        self.button_layout.addWidget(up_button)

        down_button = QPushButton('Down', self)
        down_button.setFocusPolicy(Qt.NoFocus)
        down_button.setToolTip('Moves selected component down')
        self.connect(down_button, SIGNAL('clicked()'), self.moveComponentDown)
        self.button_layout.addWidget(down_button)

        delete_button = QPushButton('Delete', self)
        delete_button.setFocusPolicy(Qt.NoFocus)
        delete_button.setToolTip('Remove selected component from model manager instance')
        self.connect(delete_button, SIGNAL('clicked()'), self.deleteComponent)
        self.button_layout.addWidget(delete_button)

        # read and save buttons are not accessible from contextual menus.
        self.read_button = QPushButton('Read', self)
        self.read_button.setFocusPolicy(Qt.NoFocus)
        self.read_button.setToolTip('Rad model from file.')
        self.button_layout.addWidget(self.read_button)

        self.save_button = QPushButton('Save', self)
        self.save_button.setFocusPolicy(Qt.NoFocus)
        self.save_button.setToolTip('Save model to file.')
        self.button_layout.addWidget(self.save_button)

        # expression text field
        self.expression_field = QLineEdit('Expression goes here blah b;ah b;ah', self)
        self.expression_field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.expression_field.setToolTip('Model expression.')
        self.expression_layout.addWidget(self.expression_field)
        self.expression_field.setFocusPolicy(Qt.NoFocus)  # remove to enable editing

        if hasattr(model, 'compound_model'):
            compound_model = model.compound_model
            self.updateExpressionField(compound_model)

        # setup to gray out buttons based on context.
        self.treeView.setButtons(up_button, down_button, delete_button, self.save_button, model)

        # connect signals.
        self.connect(self.save_button, SIGNAL('clicked()'), self.saveModel)
        self.connect(self.read_button, SIGNAL('clicked()'), self.readModel)
        self.connect(self, SIGNAL("treeChanged"), self._setSaveButtonLooks)

    # this will change the Save button appearance depending on how many
    # components are stored in the current active list. For now we don't
    # allow single components to be saved to file, since the concept
    # itself is mostly useful for saving complex compound models. It
    # remains to be seen if this assumption will hold under user scrutiny.
    def _setSaveButtonLooks(self):
        self.save_button.setEnabled(len(self.model.items) > 0)

    def updateExpressionField(self, compound_model):
        expression = ""
        if compound_model and hasattr(compound_model, '_format_expression'):
            expression = compound_model._format_expression()
        self.expression_field.setText(expression)

    # contextual menu
    def openMenu(self, position):
        self.position = position
        level, index = self.findTreeLevel()
        if index.isValid():
            menu = QMenu()
            menu.addAction(self.createAction(menu, "Delete component", self.deleteComponent))
            row = index.row()
            if row > 0:
                menu.addAction(self.createAction(menu, "Move component up", self.moveComponentUp))
            if row < self.model.rowCount() - 1:
                menu.addAction(self.createAction(menu, "Move component down", self.moveComponentDown))

            # We would use code like this in case we need contextual
            # menus at other levels in the tree besides the first level.
            #
            #            level = self.findTreeLevel()
            #            if level == 0:
            #                menu.addAction(self.createAction(menu, "Delete component", self.deleteComponent))
            #                row = self.treeView.indexAt(self.position).row()
            #                if row > 0:
            #                    menu.addAction(self.createAction(menu, "Move component up", self.moveComponentUp))
            #                if row < self.model.rowCount()-1:
            #                    menu.addAction(self.createAction(menu, "Move component down", self.moveComponentDown))
            #            elif level == 1:
            #                 placeholder for edit parameter functionality.
            #            elif level == 2:
            #                menu.addAction(self.createAction(menu, "Edit parameter value", self.editParameterValue))

            menu.exec_(self.treeView.viewport().mapToGlobal(position))

    # Callbacks for contextual menus and buttons. The 'treeChanged'
    # signal is emitted so caller code can be notified when buttons
    # and menus are activated and the trees get re-arranged.
    def deleteComponent(self):
        level, index = self.findTreeLevel()
        if level >= 0:
            self.model.takeRow(index.row())
            self.treeView.clearSelection()
            self.emit(SIGNAL("treeChanged"), index.row())

    def moveComponentUp(self):
        level, index = self.findTreeLevel()
        if level >= 0 and index.row() > 0:
            is_expanded = self.treeView.isExpanded(index)
            items = self.model.takeRow(index.row())
            self.model.insertRow(index.row() - 1, items)
            index_above = self.treeView.indexAbove(index)
            self.treeView.setExpanded(index_above, is_expanded)
            self.treeView.clearSelection()
            self.emit(SIGNAL("treeChanged"), index.row())

    def moveComponentDown(self):
        level, index = self.findTreeLevel()
        if level >= 0 and index.row() < self.model.rowCount() - 1:
            is_expanded = self.treeView.isExpanded(index)
            items = self.model.takeRow(index.row())
            self.model.insertRow(index.row() + 1, items)
            index_below = self.treeView.indexBelow(index)
            self.treeView.setExpanded(index_below, is_expanded)
            self.treeView.clearSelection()
            self.emit(SIGNAL("treeChanged"), index.row())

    def saveModel(self):
        global _model_directory # retains memory of last visited directory
        sp_model_io.saveModelToFile(self, self.model.compound_model, _model_directory)

    def readModel(self):
        global _model_directory # retains memory of last visited directory
        fname = QFileDialog.getOpenFileName(self, 'Open file', _model_directory)
        compound_model, _model_directory = sp_model_io.buildModelFromFile(fname)
        expression = ""
        if compound_model:
            if hasattr(compound_model, '_submodels'):
                for model in compound_model:
                    self.model.addOneElement(model)
            else:
                self.model.addOneElement(compound_model)

            if hasattr(compound_model, '_format_expression'):
                expression = compound_model._format_expression()

            self.emit(SIGNAL("treeChanged"), 0)

        self.expression_field.setText(expression)



# Parameter values can be edited directly from their QStandardItem
# representation. The code below (still incomplete) is an attempt
# to use contextual menus for the same purpose.
#
#    def editParameterValue(self):
#        parameter_index = self.treeView.indexAt(self.position).parent()
#        parameter_row = parameter_index.row()
#
#        function_row = parameter_index.parent().row()
#        item = self.model.item(function_row)
#        function = item.getDataItem()
#
#        for param_name in function.param_names:
#            if function._param_orders[param_name] == parameter_row:
#                break
#        parameter =  function.__getattribute__(param_name)
#
#        print "AstropyModelingTest - line 163: ",  parameter.value



# Window with the spectral component library ----------------------------------

class _SpectralLibraryGUI(object):

# Attempt to get classes directly from the models module.
# Doesn't work, need to get classes from Model registry instead.
    # def __init__(self, models_gui):
    #     data = []
    #     for key in models.__all__:
    #         function_metaclass = models.__dict__[key]
    #         if issubclass(function_metaclass, Fittable1DModel):
    #             data.append(function_metaclass)
    #
    #     self.window = LibraryWindow(data, models_gui)

    def __init__(self, models_gui, x, y, drop_down=True):
        data = []
        keys = sorted(models_registry.registry.keys())
        for key in keys:
            function = models_registry.registry[key]
            # redundant test for now, but needed in case we
            # switch to introspection from the models registry.
            # if issubclass(function.__class__, Fittable1DModel) or \
            #    issubclass(function.__class__, PolynomialModel):
            # TODO Polynomials do not carry internal instances of
            # Parameter. This makes the code in this module unusable,
            # since it relies on the existence of parameter instances
            # in the spectral model functions. To make it usable, we
            # need to add special handling code that can get and set
            # polynomial coefficients. Thus suggests that polynomials
            # were not designed to be mixed in with instances of
            # Fittable1DModel. This could make sense from a software
            # design standpoint, but it is hardly what the use cases
            # seem to imply.
            if issubclass(function.__class__, Fittable1DModel):
                data.append(function)

        self.model = SpectralComponentsModel(name=AVAILABLE_COMPONENTS)
        self.model.addItems(data)

        # Look-and-feel can be based either on a split pane or a drop down menu.
        if drop_down:
            self.window = _LibraryComboBox(self.model, models_gui, x, y)
        else:
            self.window = _LibraryWindow(self.model, models_gui, x, y)

    def getSelectedModel(self):
        return self.window.getSelectedModel()

    def setArrays(self, x, y):
        self.window.setArrays(x, y)


class _LibraryWindow(_BaseWindow):
    def __init__(self, model, models_gui, x, y):
        super(_LibraryWindow, self).__init__(model)
        self.models_gui = models_gui

        # numpy arrays used to instantiate functions.
        self.x = x
        self.y = y

        # Contextual menus do not always work under ipython
        # non-block mode. The Add button is an alternative
        # way of implementing the same action.
        add_button = QPushButton('Add', self)
        add_button.setFocusPolicy(Qt.NoFocus)
        add_button.setToolTip('Adds selected component to active model')
        self.connect(add_button, SIGNAL('clicked()'), self.addComponent)
        self.button_layout.addWidget(add_button)

    # callback for the Add button
    def addComponent(self):
        function = self.getSelectedModel()

        sp_adjust.adjust(function, self.x, self.y)

        self._addComponentToActive(function)
        self.treeView.clearSelection()

    # contextual menu.
    def openMenu(self, position):
        index = self.treeView.indexAt(position)
        if index.isValid():
            menu = QMenu()
            menu.addAction(self.tr("Add component"))
            menu.exec_(self.treeView.viewport().mapToGlobal(position))
            # no need to add an action to this menu since it has only one
            # element. Just do the action straight away.
            item = self.model.item(index.row())
            if item:
                function = item.getDataItem()
                self._addComponentToActive(function)

        self.treeView.clearSelection()

        # This is an attempt to instantiate from the class registry.
        #
        # param_names = inspect.getargspec(meta.__init__)[0]
        # param_values = np.ones(len(param_names)-1)
        #
        # inst = models_registry[name](param_values)
        #
        # cls = type.__new__(type(meta), name, (Fittable1DModel,), {'param_names': param_names[1:]})
        # cls = type(name, (Fittable1DModel,), {'param_names': param_names[1:]})
        #
        # args = {}
        # i = 0
        # for pn in param_names[1:]:
        #     args[pn] = param_values[i]
        #     i += 1
        #
        # inst = cls.__init__(**args)

    def setArrays(self, x, y):
        self.x = x
        self.y = y

    # Adds the selected spectral model component to the active model.
    def _addComponentToActive(self, component):
        name = models_registry.get_component_name(component)

        self.finalizeAddingComponent(name)

    def finalizeAddingComponent(self, name):
        # this should perhaps be done by instantiating from a
        # class. We instead resort to brute force and copy the
        # instance instead. It works.....
        if name in models_registry.registry:
            component = models_registry.registry[name].copy()
            if component:
                sp_adjust.adjust(component, self.x, self.y)
                self.models_gui.updateModel(component)


class _LibraryComboBox(QComboBox, _LibraryWindow):
    def __init__(self, model, models_gui, x, y):
        QComboBox.__init__(self)
        _LibraryWindow.__init__(self, model, models_gui, x, y)

        self.addItem(AVAILABLE_COMPONENTS)

        nrows = self.model.rowCount()
        for i in range(nrows):
            index = self.model.index(i, 0)
            data = self.model.data(index)

            self.addItem(data.toString())

        self.activated.connect(self._addSelectedComponent)

    # Adds the selected spectral components.
    def _addSelectedComponent(self):
        name = str(self.currentText())
        self.finalizeAddingComponent(name)


# The MVC Model classes -----------------------------------------

# Item classes

# Base item is a QStandardItem with the ability to directly
# hold a reference to the spectral object being represented
# in the tree.

class SpectralComponentItem(QStandardItem):
    def __init__(self, name):
        QStandardItem.__init__(self)
        if name is None:
            name = "None"
        self.setData(name, role=Qt.DisplayRole)
        self.setEditable(False)

    def setDataItem(self, item):
        self.item = item

    def getDataItem(self):
        return self.item


# Value item specializes the base item to make it editable.
# or checkable. The slot connected to the tree model's
# itemChanged signal must be able to differentiate among the
# several possible items, using the 'type' attribute and the
# 'isCheckable' property.

class SpectralComponentValueItem(SpectralComponentItem):
    def __init__(self, parameter, type, checkable=False, editable=True):
        self.parameter = parameter
        self.type = type
        # boolean attributes display attribute
        # value via a checkbox, not text.
        id_str = type
        if not checkable:
            id_str = type + ": " + str(getattr(self.parameter, type))
        SpectralComponentItem.__init__(self, id_str)
        self.setEditable(editable)
        self.setCheckable(checkable)
        # checkbox setup.
        if checkable and getattr(self.parameter, type):
            self.setCheckState(Qt.Checked)


# Tied item specializes the base item to handle the specifics
# of a callable tie. The slot connected to the tree model's
# itemChanged signal must be able to differentiate among the
# several possible items, using the 'type' attribute and the
# 'isCheckable' property. This is not necessary for now, since
# this item type is being defined as non-editable. For now, the
# only way for the user to modify a tie is to directly edit an
# importable file with the model definition.

class SpectralComponentTiedItem(SpectralComponentItem):
    def __init__(self, parameter):
        self.parameter = parameter
        self.type = "tied"

        tie = getattr(self.parameter, self.type)
        id_str = self.type + ": " + sp_model_io.get_tie_text(tie)

        SpectralComponentItem.__init__(self, id_str)
        self.setEditable(False)  # for now!!
        self.setCheckable(False)


# Model classes

# This class provides the base model for both the active
# and the library windows. The base model handles the
# tree's first level, where the component names are held.

class SpectralComponentsModel(QStandardItemModel):
    def __init__(self, name):
        QStandardItemModel.__init__(self)
        self.setHorizontalHeaderLabels([name])

    def addItems(self, elements):
        if hasattr(elements, '__getitem__'):
            for element in elements:
                self.addOneElement(element)
        else:
            self.addOneElement(elements)

    def addOneElement(self, element):
        name = models_registry.get_component_name(element)
        self.addToModel(name, element)

    def addToModel(self, name, element):
        item = SpectralComponentItem(name)
        item.setDataItem(element)
        parent = self.invisibleRootItem()
        parent.appendRow(item)


# RE pattern to decode scientific and floating point notation.
_pattern = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

def _float_check(value):
    """ Checks for a valid float in either scientific or floating point notation"""
    substring = _pattern.findall(str(value))
    if substring:
        number = float(substring[0])
        if len(substring) > 1:
            number *= math.pow(10., int(substring[1]))
        return number
    else:
        return False


# This class adds to the base model class the ability to handle
# two additional tree levels. These levels hold respectively
# the parameter names of each component, and each parameter's
# editable attributes.

class ActiveComponentsModel(SpectralComponentsModel):

    def __init__(self, components, name):
        SpectralComponentsModel.__init__(self, name)

        if components:
            self.compound_model = components
            self.addItems(self.compound_model)

        self.itemChanged.connect(self._onItemChanged)

    # TODO use QDataWidgetMapper
    # this violation of MVC design principles is necessary
    # so our model manager class can work with the modified
    # code in modelmvc.py. This probably could be done via
    # the QDataWidgetMapper stuff instead.
    def setWindow(self, window):
        self._window = window

    def addToModel(self, name, element):
        # add component to tree root
        if element.name:
            item = SpectralComponentItem(name + " (" + str(element.name) + ")")
        else:
            item = SpectralComponentItem(name)
        item.setDataItem(element)
        parent = self.invisibleRootItem()
        parent.appendRow(item)

        nameItem = SpectralComponentValueItem(element, "name")
        nameItem.setDataItem(element.name)
        # nameItem.setEditable(True)
        item.appendRow(nameItem)

        # now add parameters to component in tree.
        for e in element.param_names:
            par = element.__getattribute__(e)
            if isinstance(par, Parameter):

                # add parameter. Parameter name is followed
                # by its value when displaying in tree.
                parItem = SpectralComponentItem(par.name + ": " + str(par.value))
                parItem.setDataItem(par)
                item.appendRow(parItem)

                # add parameter value and other attributes to parameter element.
                valueItem = SpectralComponentValueItem(par, "value")
                valueItem.setDataItem(par.value)
                parItem.appendRow(valueItem)
                minItem = SpectralComponentValueItem(par, "min")
                minItem.setDataItem(par.min)
                parItem.appendRow(minItem)
                maxItem = SpectralComponentValueItem(par, "max")
                maxItem.setDataItem(par.max)
                parItem.appendRow(maxItem)
                fixedItem = SpectralComponentValueItem(par, "fixed", checkable=True)
                fixedItem.setDataItem(par.fixed)
                parItem.appendRow(fixedItem)
                tiedItem = SpectralComponentTiedItem(par)
                tiedItem.setDataItem(par.tied)
                parItem.appendRow(tiedItem)

    @property
    def items(self):
        result = []
        for i in range(self.rowCount()):
            result.append(self.item(i).item)
        return result

    def _floatItemChanged(self, item):
        type = item.type
        number = _float_check(item.text())
        if number:
            if hasattr(item, 'parameter'):
                setattr(item.parameter, type, number)
                item.setData(type + ": " + str(number), role=Qt.DisplayRole)
                # parameter name is followed by its value when displaying in tree.
                if type == 'value':
                    item.parent().setData(item.parameter.name + ": " + str(number), role=Qt.DisplayRole)
        else:
            item.setData(type + ": " + str(getattr(item.parameter, type)), role=Qt.DisplayRole)

    def _nameChanged(self, item):
        old_name = item.item
        new_name = str(item.text())
        # remove actual new name from the "name:newname" in the tree display
        index = new_name.find(":")
        if index > -1:
            new_name = new_name[index+2:]
        if len(new_name) > 0:
            setattr(item, "item", new_name)
            if index <= -1:
                item.setData("name: " + new_name, role=Qt.DisplayRole)
            # function name is followed by component name when displaying on tree
            item_parent = item.parent()
            setattr(item_parent.item, "_name", new_name)
            id_string = str(item_parent.text())
            index = id_string.find("(")
            function_name = id_string[:index-1]
            item_parent.setData(function_name + " (" + new_name + ")", role=Qt.DisplayRole)

            # name was successfully changed; now check to see if any tied parameters depend om it.
            self._modify_tied_components(item, old_name, new_name)

        else:
            item.setData("name: " + old_name, role=Qt.DisplayRole)

    def _booleanItemChecked(self, item):
        setattr(item.parameter, item.type, (item.checkState() == Qt.Checked))

    def _onItemChanged(self, item):
        if item.isCheckable():
            self._booleanItemChecked(item)
        elif item.type == "name":
            self._nameChanged(item)
        elif item.type in ("value", "min", "max"):
            self._floatItemChanged(item)

    # scans all parameters in all components in the model, looking for
    # tied parameters that point to the old name. Replace the old name
    # with the new name in the tie. This assumes that we use the standard
    # lambda form for ties.
    def _modify_tied_components(self, reference_item, old_name, new_name):
        for row, component in enumerate(self.items):
            if component.tied:
                row2 = 1
                for key, tie in component.tied.items():
                    row2 += 1
                    if tie:
                        tie_text = sp_model_io.get_tie_text(tie)
                        if old_name in tie_text:

                            # modify actual component
                            new_tie_text = tie_text.replace(old_name, new_name)
                            new_tie = eval(new_tie_text)
                            component.tied[key] = new_tie

                            # modify element in tree
                            item = self.item(row)
                            # sometimes the tree returns a None item.
                            if item:
                                tie_element = item.child(row2).child(4)
                                text = tie_element.text()
                                new_text = text.replace(old_name, new_name)
                                tie_element.setData(new_text, role=Qt.DisplayRole)


class SpectralModelManager(QObject):
    """ Basic class to be called by external code.

    It is responsible for building the GUI trees and putting them together
    into a split pane layout. An alternate, single pane plus drop-down
    menu, is also available. The class also provides accessors to the active
    model individual spectral components and to the library functions,
    as well as to the spectrum that results from a compound model call.

    It inherits from QObject for the sole purpose of being able to
    respond to Qt signals.

    Parameters
    ----------
    model: list or string or variable, optional
      List with instances of spectral components from
      astropy.modeling.functional_models. If not provided,
      the instance will be initialized with an empty list.
      Or it can be a string with a fully specified file name
      which contains a compound model specification. Or, it
      can be a Python reference to a compound model instance.

    drop_down: boolean, optional
      Defines GUI looks. Default is True, meaning that the available
      spectral components from the astropy.modeling.models library
      are accessed via a drop down menu. If set to False, the
      components are accessed from a separate tree on a split pane
      window.

    """
    def __init__(self, model=None, drop_down=True):
        super(SpectralModelManager, self).__init__()

        # _init_compound_model is used just to hold a reference
        # to any compound model one wishes to use to start up
        # the tool. The actual compound model used in operations
        # is set by the buildMainPanel method. It lives in
        # self.model_gui.model.compound_model.
        if model == None:
            self._init_compound_model = None
        elif type(model) == type(list):
            self._init_compound_model = _buildSummedCompoundModel(model)
        elif type(model) == type(""):
            global _model_directory
            self._init_compound_model, _model_directory = sp_model_io.buildModelFromFile(model)
        else:
            self._init_compound_model = model

        self._drop_down = drop_down

        self.x = None
        self.y = None

        self.changed = SignalModelChanged()
        self.selected = SignalComponentSelected()

    def setArrays(self, x, y):
        ''' Defines the region in spectral coordinate vs. flux
        'space' to which the components in the model should refer
        to.

        For now, this region is being defined by the data arrays
        associated with the observational data at hand. The region
        could conceivably be defined by any other means, as long
        as the functional components can then use the region data
        to initialize their parameters with sensible values.

        This region is used by code in module sp_adjust. If no
        X and/or Y arrays are provided via this method, spectral
        components added to the compound model will be initialized
        to a default set of parameter values.
        
        Parameters
        ----------
        x: numpy array
          Array with spectral coordinates
        y: numpy array
          Array with flux values

        '''
        self.x = x
        self.y = y

        if  hasattr(self, '_library_gui'):
            self._library_gui.setArrays(self.x, self.y)

    def buildMainPanel(self, model=None):
        """ Builds the main panel with the active and the library
        trees of spectral components.

        Parameters
        ----------
        model: list or str, optional
          List with instances of spectral components from
          astropy.modeling.functional_models. Or, a file name
          in the 'specfit' format from where a compound model
          can be imported. If not provided, the list of components
          will exist but will be empty.

        Returns
        -------
          instance of either QMainWindow or QSplitter

        """
        # override whatever model was passed to the constructor.
        # This specific form of the conditional avoids a mishap
        # when self._init_compound_model is an empty list.
        if model == None:
            self._init_compound_model = None
        elif type(model) == type(list):
            self._init_compound_model = _buildSummedCompoundModel(model)
        elif type(model) == type(""):
            global _model_directory
            self._init_compound_model, _model_directory = sp_model_io.buildModelFromFile(model)
        else:
            self._init_compound_model = model

        # When called the first time, build the two trees.
        # Subsequent calls must re-use the existing trees
        # so as to preserve user selections and such.
        if not hasattr(self, 'models_gui'):
            # note that _init_compound_model is passed as an initializer, but
            # any other reference to the actual compound model that lives in
            # the GUI must be done via reference self.models_gui.model.compound_model.
            self.models_gui = _SpectralModelsGUI(self._init_compound_model)
            self._library_gui = _SpectralLibraryGUI(self.models_gui, self.x, self.y, drop_down=self._drop_down)

        if self._drop_down:
            # window contains the active tree in the central
            # widget, and the library tree in the menu widget.
            main_widget = QMainWindow();
            main_widget.setMenuWidget(self._library_gui.window)
            main_widget.setCentralWidget(self.models_gui.window)
        else:
            # split window contains the active tree in the first
            # pane and library tree in the second pane.
            main_widget = QSplitter();
            main_widget.addWidget(self.models_gui.window)
            main_widget.addWidget(self._library_gui.window)
            main_widget.setStretchFactor(0, 1)
            main_widget.setStretchFactor(1, 0)

        # Data change and click events must be propagated to the outside world.
        self.connect(self.models_gui.window, SIGNAL("treeChanged"), self._broadcastChangedSignal)
        self.connect(self.models_gui.window.treeView, SIGNAL("dataChanged"), self._broadcastChangedSignal)
        self.models_gui.window.treeView.clicked.connect(self._broadcastSelectedSignal)

        return main_widget

    def _broadcastChangedSignal(self):
        self.changed()

    def _broadcastSelectedSignal(self):
        self.selected()

    @property
    def treeWidget(self):
        """ Accessor to the tree with rendering of active spectral components.

        Returns
        -------
          instance of QTreeView

        """
        return self.models_gui.window.treeView

    @property
    def components(self):
        """ Accessor to the list with active spectral components.

        Returns
        -------
          instance of list

        """
        return self.models_gui.model.items

    def spectrum(self, wave):
        ''' Computes the compound model flux values,
        given an array of spectral coordinate values.

        Parameters
        ----------
        wave: numpy array
          Array with spectral coordinate values.

        Returns
        -------
        A numpy array with flux values. If no components exist in
        the model, a zero-valued array is returned instead.

        '''
        # The compound_model can be either a list of components
        # or a compound model instance. In the case of a
        # list, we just add the components sequentially.
        if len(self.components) > 0:
            if not type(self.models_gui.model.compound_model) == type([]):
                compound_model = self.models_gui.model.compound_model
            else:
                compound_model = _buildSummedCompoundModel(self.components)

            return compound_model(wave)

        else:
            return np.zeros(len(wave))

    def addComponent(self, component):
        ''' Adds a new spectral component to the manager.

        Parameters
        ----------
        component: astropy.modeling.Fittable1DModel
          The component to be added to the manager.

        '''
        component = sp_adjust.adjust(component, self.x, self.y)
        self.models_gui.updateModel(component)

    def getSelectedFromLibrary(self):
        ''' Returns component instance prototype selected in the
        library window. Without

        Returns
        -------
          model selected in the library window.

        '''
        return self._library_gui.getSelectedModel()

    def selectedComponent(self):
        ''' Returns component selected in the active components window.

        Returns
        -------
          component selected in the active components window.

        '''
        return self.models_gui.getSelectedModel()

    def modifyModel(self, new_components):
        ''' Replaces spectral components with new instances.

        This method must be called with a list of components that
        matches the existing components in the model manager active
        list. The method's purpose is to replace the parameter
        values of each component with the values of the paired
        component in the input list. Thus the lists have to match
        perfectly.

        Parameters
        ----------
        new_components: list
          list with instances of astropy.modeling.Fittable1DModel
          to be added to the manager.

        '''
        for i, c in enumerate(self.components):
            nc = new_components[i]
            c.parameters = nc.parameters

            # modify the tree model so the fit results
            # show immediately on the display.
            for j, value in enumerate(c.parameters):
                item = self.models_gui.model.item(i).child(j).child(0)
                item.setData("value: " + str(value), role=Qt.DisplayRole)




class SignalModelChanged(signal_slot.Signal):
    ''' Signals that a change in the model took place. '''

class SignalComponentSelected(signal_slot.Signal):
    ''' Signals that a component has been selected. '''

