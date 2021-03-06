import os, sys, re, dis

from cStringIO import StringIO
from PyQt4.QtGui import QFileDialog

import models_registry


# Builds a compound model specified in a .py file
def buildModelFromFile(fname):
    directory = os.path.dirname(str(fname))
    sys.path.append(directory)
    f = os.path.basename(str(fname)).split('.')[0] # remove .py from end of file name so it can be imported

    # if already defined, force a reload.
    import_statement = "import " + f
    if f in sys.modules.keys():
        import_statement = "reload(sys.modules['" + f + "'])"

    try:
        exec import_statement
        module = sys.modules[f]

        # this will pick up the first model defined in the file. A different
        # mechanism is needed to handle files with multiple model definitions.
        for variable in dir(module):
            if not variable.startswith('__'):
                # this assumes that the module contains only model definitions and
                # imports for the functional types used in the model definitions.
                # This 'if' statement skips the function types, everything that
                # passes is assumed to be a valid compound model definition.
                if (str(type(module.__dict__[variable]))).find('astropy.modeling.core._ModelMeta') < 0:
                    compound_model = module.__dict__[variable]
                    return compound_model, directory
        return None,None
    except Exception as e:
        print("ERROR: " + str(e))
        return None,None


# Builds a model expression inside a string, and dumps string to file.
def saveModelToFile(parent, model, model_directory):
    if hasattr(model, '_format_expression'):
        expression_string, prolog = _buildCompoundModelExpression(model)
    else:
        expression_string, prolog = _buildSingleComponentExpression(model)

    # Write to file.
    fname = QFileDialog.getSaveFileName(parent, 'Write to file', model_directory)

    if len(fname) > 0:
        f = os.open(fname, os.O_RDWR|os.O_CREAT)
        os.write(f, prolog)
        os.write(f, expression_string)
        os.close(f)


def _buildSingleComponentExpression(model):
    name = models_registry.get_component_name(model)
    path = models_registry.get_component_path(model)

    prolog = "from " + path + " import " + name + "\n\n"
    expression_string = "model1 = \\\n" + _assemble_component_spec(model)

    return expression_string, prolog


def _buildCompoundModelExpression(model):
    # The following assumes that the formatted string expression
    # in an astropy compound model has operands of the form [0], [1],
    # etc, that is, a sequential number enclosed in square brackets.
    expression = model._format_expression()
    tokens = re.split(r'[0-9]+', expression)
    # this loop builds the main expression, and captures
    # information needed for building the file header (where
    # the import statements go).
    expression_string = ""
    import_module_names = {}
    for token, component in zip(tokens, model):
        # clean up astropy-inserted characters
        token = token.replace('[', '')
        token = token.replace(']', '')

        expression_string += str(token) + _assemble_component_spec(component)

        # here we store the module paths for each component. Using
        # a dictionary key ensures that we get only one of each.
        path = models_registry.get_component_path(component)
        name = models_registry.get_component_name(component)
        import_module_names[name] = path

    # this loop now uses the captured information from above to
    # build a set of import statements that go at the beginning
    # of the file.
    prolog = ""
    for name, path in import_module_names.iteritems():
        prolog += "from " + path + " import " + name + "\n"
    prolog += "\n"
    # we need to add a reference to the model so it can actually
    # be used after imported. We just use 'model1' for the variable
    # name. This also implicitly assumes that only one model will be
    # stored in the file. It remains to be seen how useful this
    # assumption will be in practice.
    prolog += "model1 = \\\n"
    return expression_string, prolog


# Disassembles a tie callable. Ties read from a model
# file are not directly accessible in text form because
# the model file is compiled at import time.
def get_tie_text(tie):
    if tie:
        # dis() only outputs on standard output.....
        keep = sys.stdout
        sys.stdout = StringIO()
        dis.dis(tie)
        assembler_text = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = keep
        result = _parse_assembler_text(assembler_text)
    else:
        result = 'False'
    return result


# This parses the text returned by the disassembler for
# a lambda function that multiplies a constant by a
# variable. That is, we are assuming that ties are coded
# as lambda functions with multiplication by a constant,
# as in STSDAS' specfit.
parser = re.compile(r'\(([^)]+)\)') # picks up whatever is enclosed in parenthesis
def _parse_assembler_text(text):
    tokens = parser.findall(text)
    factor = tokens[0]
    lambda_variable_name = tokens[1]
    function_id = tokens[2]
    par_name = tokens[3]

    return "lambda %s: %s *  %s[%s].%s" % \
           (lambda_variable_name,
            factor,
            lambda_variable_name,
            function_id,
            par_name)


# Builds an operand (a spectral component) for an astropy compound model.
def _assemble_component_spec(component):
    result = ""

    # function name - Note that get_component_name works
    # pretty much independently of the models registry.
    # Any model will work because the function name is
    # derived from the component's __class__.
    name = models_registry.get_component_name(component)
    result += name

    # component name
    if component.name:
        result += "(name = \'"
        result += component.name + "\',\n"
    else:
        result += "(\n"

    # parameter names and values
    for i, param_name in enumerate(component.param_names):
        result += "            " + param_name
        result += " = "
        result += str(component.parameters[i]) + ",\n"

    # parameter bounds
    bounds = component.bounds
    result += "            bounds = {\n"
    for param_name in component.param_names:
        result += "                     '" + param_name + "': " + str(bounds[param_name]) + ",\n"
    result += "                     },\n"

    # parameter fixed flags
    fixed = component.fixed
    result += "            fixed = {\n"
    for param_name in component.param_names:
        result += "                     '" + param_name + "': " + str(fixed[param_name]) + ",\n"
    result += "                    },\n"

    # parameter ties. Ties have to be disassembled and parsed
    # in order to become human-readable and writable to file.
    ties = component.tied
    result += "            tied = {\n"
    for param_name in component.param_names:
        tie_text = get_tie_text(ties[param_name])
        result += "                    '" + param_name + "': " + tie_text + ",\n"
    result += "                   },\n"

    result += "            ) \\\n"
    return result



