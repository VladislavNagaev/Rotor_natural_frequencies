import eel
from lib import *
from pandas import DataFrame


@eel.expose
def data_path_return(kwargs):

    data_folder = kwargs.get('data_folder')
    data_path = kwargs.get('data_path')

    # Относительная директория файла
    dirname_rotor_data_path = os.path.dirname(data_path)
    # Наименование файла
    basename_rotor_data_path = os.path.basename(data_path)

    if dirname_rotor_data_path == '':
        full_rotor_data_path = os.path.join(data_folder, basename_rotor_data_path)
    else:
        full_rotor_data_path = os.path.abspath(data_path)

    return full_rotor_data_path


@eel.expose
def param_checker(kwargs):

    try:
        data = Variables.parse_obj(kwargs).dict(by_alias=True)
        status = True
    except ValidationError as e:
        data = dict()
        status = False

    param_name = list(kwargs.keys())[0]

    return {'value':data.get(param_name), 'status':status}


@eel.expose
def load_rotor_data(kwargs):
    global rotor
    rotor = Rotor()

    try:
        data = Variables.parse_obj(kwargs).dict(by_alias=True)
    except ValidationError as e:
        return False

    return rotor.load_rotor_data(
        rotor_data_path=data.get('rotor_data_path'),
        data_format=data.get('data_format'),
        data_encoding=data.get('data_encoding'),
        data_sep=data.get('data_sep'),
        data_decimal=data.get('data_decimal'),
    ) 


@eel.expose
def return_rotor_data():
    global rotor

    try:
        rotor_data = rotor.rotor_data
    except Exception:
        rotor_data = None
    
    if isinstance(rotor_data, DataFrame):
        data_values = rotor_data.fillna(value='', inplace=False).T.to_dict()
        data_index = rotor_data.index.to_list()
        data_columns = rotor_data.columns.to_list()
        rotor_data = dict(data_values=data_values, data_index=data_index, data_columns=data_columns)

    return rotor_data


@eel.expose
def return_support_data():
    global rotor

    try:
        support_data = rotor.support_data
    except Exception:
        support_data = None
    
    if isinstance(support_data, DataFrame):
        data_values = support_data.fillna(value='', inplace=False).T.to_dict()
        data_index = support_data.index.to_list()
        data_columns = support_data.columns.to_list()
        support_data = dict(data_values=data_values, data_index=data_index, data_columns=data_columns)

    return support_data


@eel.expose
def calculate_result_data(kwargs):
    global rotor

    try:
        data = Variables.parse_obj(kwargs).dict(by_alias=True)
    except ValidationError as e:
        return False

    return rotor.calculate_result_data(
        natural_frequencies_number=data.get('natural_frequencies_number'),
        F1=data.get('F1'),
        FT1=data.get('FT1'),
        fk=data.get('fk'),
        EPSF=data.get('EPSF'),
        normalize=data.get('normalize'),
    )


@eel.expose
def return_result_data():
    global rotor

    try:
        result = rotor.result
    except Exception:
        result = None

    if isinstance(result, DataFrame):
        result = result.shape

    return result


@eel.expose
def return_stiffness_data():
    global rotor

    try:
        stiffness = rotor.stiffness
    except Exception:
        stiffness = None
    
    if isinstance(stiffness, DataFrame):
        stiffness = stiffness.shape

    return stiffness


@eel.expose
def save_result_data(kwargs):
    global rotor

    try:
        data = Variables.parse_obj(kwargs).dict(by_alias=True)
    except ValidationError as e:
        return False

    return rotor.save_result_data(
        data_folder=data.get('data_folder'),
        format=data.get('data_format'),
        encoding=data.get('data_encoding'),
        sep=data.get('data_sep'),
        decimal=data.get('data_decimal'),
    )


@eel.expose
def save_stiffness_data(kwargs):
    global rotor

    try:
        data = Variables.parse_obj(kwargs).dict(by_alias=True)
    except ValidationError as e:
        return False
    
    return rotor.save_stiffness_data(
        data_folder=data.get('data_folder'),
        format=data.get('data_format'),
        encoding=data.get('data_encoding'),
        sep=data.get('data_sep'),
        decimal=data.get('data_decimal'),
    )


@eel.expose
def prepare_visualizations1(kwargs):
    global rotor
    return rotor.prepare_visualizations1(
        stiffness_on_plot=kwargs.get('stiffness_on_plot'),
        plot_dpi=kwargs.get('plot_dpi'),
        plot_format=kwargs.get('plot_format'),
        frequency_units=kwargs.get('frequency_units'),
    )


@eel.expose
def prepare_visualizations2(kwargs):
    global rotor
    return rotor.prepare_visualizations2(
        plot_dpi=kwargs.get('plot_dpi'),
        plot_format=kwargs.get('plot_format'),
        frequency_units=kwargs.get('frequency_units'),
        horizontal_boundaries=kwargs.get('horizontal_boundaries'),
        vertical_boundaries=kwargs.get('vertical_boundaries'),
        restricted_areas=kwargs.get('restricted_areas'),
        restricted_areas_width=kwargs.get('restricted_areas_width'),
    )


@eel.expose
def return_visualizations1():
    global rotor

    try:
        visualizations = rotor.visualizations1
    except Exception:
        visualizations = None

    return visualizations


@eel.expose
def return_visualizations2():
    global rotor

    try:
        visualizations = rotor.visualizations2
    except Exception:
        visualizations = None

    return visualizations


@eel.expose
def save_visualizations1(kwargs):
    global rotor

    try:
        data = Variables.parse_obj(kwargs).dict(by_alias=True)
    except ValidationError as e:
        return False

    return rotor.save_visualizations1(data_folder=data.get('data_folder'))


@eel.expose
def save_visualizations2(kwargs):
    global rotor

    try:
        data = Variables.parse_obj(kwargs).dict(by_alias=True)
    except ValidationError as e:
        return False

    return rotor.save_visualizations2(data_folder=data.get('data_folder'))



if __name__ == "__main__":

    rotor = Rotor()

    eel.init("web")
    eel.start(
        "templates/main.html",
        # disable_cache=True,
        jinja_templates='templates',
        allowed_extensions=['.js', '.html', '.txt', '.htm', '.xhtml'],
        host='localhost',
        port=8000,
        # size=(400,600),
        mode="chrome",
        # mode="edge",
    )
