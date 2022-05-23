import os
import eel
from pandas import read_csv


def load_data(data_path:str):

    # Загрузка данных
    try:
        data = read_csv(data_path, sep=';', engine='c', decimal=',', encoding='utf-8')
    except FileNotFoundError:
        raise ValueError("Файл данных ротора по указанному пути не существует")
    
    return data


def get_current_dir():
    current_dir = os.path.curdir
    current_dir = os.path.abspath(current_dir)
    return current_dir


def get_init_params():

    global varibales_dict

    params_path = 'params.csv'

    # Проверка существования файла
    if os.path.exists(params_path):
        # Загрузка данных
        params_data = load_data(data_path=params_path)
        params_data = params_data[(params_data['feature'].notnull())&(params_data['value'].notnull())][['feature','value']]
        features = params_data['feature'].values
        value = params_data['value'].values
        params_dict = dict(zip(features,value))

        for feature, value in params_dict.items():
            for param in varibales_dict.keys():
                if varibales_dict.get(param).get('feature') == feature:
                    
                    varibales_dict[param]['value'] = value
    


param1 = dict(
    feature = 'data_folder',
    description = (
        'Рабочая директория'
        '\n\nДиректория, используемая для загрузки данных ротора и выгрузки результатов расчета.'
        '\n\nЗначение по-умолчанию: текущая директория'
        '\n\n'
    ),
    placeholder = get_current_dir(),
    value='',
    datalist = [],
)

param2 = dict(
    feature = 'rotor_data_path',
    description = (
        'Файл данных ротора'
        '\n\nНаименование файла данных ротора, если последний находится в рабочей директории, иначе абсолютный путь к файлу данных.'
        '\n\nЗначение по-умолчанию: rotor_data.xlsx'
        '\n\nТребования к файлу данных:'
        '\n\tформат - csv или xlsx'
        '\n\tпервая строка файла - строка наименований столбцов'
        '\n\tтребуемые наименования столбцов:'
        '\n\t\tsection - массив номеров участков'
        '\n\t\tlength - массив длин участков, м'
        '\n\t\tweight - массив масс участков, кг'
        '\n\t\tstiffness - массив жесткостей участков, (Н•м)'
        '\n\t\tsupport - массив участков с опорами'
        '\n\t\tstiffness_array-n - массив жесткостей опор валопровода (Н/м.*10**(-9))'
        '\n\nПримечание: количество столбцов stiffness_array может быть любым, но не менее одного. '
        'Каждый отдельный столбец будет интерпретирован как отдельный расчетный случай. '
        'Наименование каждого расчетного случая следует указывать вместо переменой "n". '
        'Например, для рассчетного случая с наимнованием "Первый_А2" следует указать наименование столбца "stiffness_array-Первый_А2". '
        'Наименование расчетных случаев используется в качестве легенды при построении графиков.'
        '\n\n'
    ),
    placeholder='rotor_data.xlsx',
    value='',
    datalist = [],
)

param3 = dict(
    feature = 'natural_frequencies_number',
    description = (
        'Число собственных частот, подлежащих отысканию'
        '\n\nНаибольшее значение собственной частоты, до которой будет выполнен рассчет. '
        'Например, для значения 5 будет выполнен расчет собственных частот от 1 до 5 включительно.'
        '\n\nФормат ввода: целое число.'
        '\n\nЗначение по-умолчанию: 5'
        '\n\n'
    ),
    placeholder='5',
    value='',
    datalist = [
        {'value':'1', 'description':'Первая собственная частота'},
        {'value':'2', 'description':'Первая - вторая собственные частоты'},
        {'value':'3', 'description':'Первая - третья собственные частоты'},
        {'value':'4', 'description':'Первая - четвертая собственные частоты'},
        {'value':'5', 'description':'Первая - пятая собственные частоты'},
        {'value':'6', 'description':'Первая - шестая собственные частоты'},
        {'value':'7', 'description':'Первая - седьмая собственные частоты'},
        {'value':'8', 'description':'Первая - восьмая собственные частоты'},
    ],
)

param4 = dict(
    feature = 'F1',
    description = (
        'Начальная частота диапзаона поиска собственных частот в Гц'
        '\n\nФормат ввода: число с плавающей точкой или целое число.'
        '\n\nЗначение по-умолчанию: 20.0'
        '\n\n'
    ),
    placeholder='20.0',
    value='',
    datalist = [],
)

param5 = dict(
    feature = 'FT1',
    description = (
        'Шаг по частоте в Гц'
        '\n\nФормат ввода: число с плавающей точкой или целое число.'
        '\n\nЗначение по-умолчанию: 0.5'
        '\n\n'
    ),
    placeholder='0.5',
    value='',
    datalist = [],
)

param6 = dict(
    feature = 'fk',
    description = (
        'Верхняя граница диапазона поиска в Гц'
        '\n\nФормат ввода: число с плавающей точкой или целое число.'
        '\n\nЗначение по-умолчанию: 1e4'
        '\n\n'
    ),
    placeholder='1e4',
    value='',
    datalist = [],
)

param7 = dict(
    feature = 'EPSF',
    description = (
        'Точность поиска частоты в Гц'
        '\n\nФормат ввода: число с плавающей точкой или целое число.'
        '\n\nЗначение по-умолчанию: 1e-7'
        '\n\n'
    ),
    placeholder='1e-7',
    value='',
    datalist = [],
)

param8 = dict(
    feature = 'normalize',
    description = (
        'Нормирование форм'
        '\n\nДопустимые значения:'
        '\n\tTrue - включить нормирование'
        '\n\tFalse - отключить нормирование'
        '\n\nЗначение по-умолчанию: True'
        '\n\n'
    ),
    placeholder='True',
    value='',
    datalist = [
        {'value':'True', 'description':'Нормировать формы'},
        {'value':'False', 'description':'Не нормировать формы'}
    ],
)

param9 = dict(
    feature = 'plot_format',
    description = (
        'Формат сохранения графиков'
        '\n\nДопустимые значения:'
        '\n\tps - Postscript'
        '\n\teps - Encapsulated Postscript'
        '\n\tpdf - Portable Document Format'
        '\n\tpgf - PGF code for LaTeX'
        '\n\tpng - Portable Network Graphics'
        '\n\traw - Raw RGBA bitmap'
        '\n\trgba - Raw RGBA bitmap'
        '\n\tsvg - Scalable Vector Graphics'
        '\n\tsvgz - Scalable Vector Graphics'
        '\n\tjpg - Joint Photographic Experts Group'
        '\n\tjpeg - Joint Photographic Experts Group'
        '\n\ttif - Tagged Image File Format'
        '\n\ttiff - Tagged Image File Format'
        '\n\nЗначение по-умолчанию: png'
        '\n\n'
    ),
    placeholder='png',
    value='',
    datalist = [
        {'value':'ps', 'description':'Postscript'},
        {'value':'eps', 'description':'Encapsulated Postscript'},
        {'value':'pdf', 'description':'Portable Document Format'},
        {'value':'pgf', 'description':'PGF code for LaTeX'},
        {'value':'png', 'description':'Portable Network Graphics'},
        {'value':'raw', 'description':'Raw RGBA bitmap'},
        {'value':'rgba', 'description':'Raw RGBA bitmap'},
        {'value':'svg', 'description':'Scalable Vector Graphics'},
        {'value':'svgz', 'description':'Scalable Vector Graphics'},
        {'value':'jpg', 'description':'Joint Photographic Experts Group'},
        {'value':'jpeg', 'description':'Joint Photographic Experts Group'},
        {'value':'tif', 'description':'Tagged Image File Format'},
        {'value':'tiff', 'description':'Tagged Image File Format'},
    ],
)

param10 = dict(
    feature = 'stiffness_on_plot',
    description = (
        'Таблица жесткостей по опорам на графике форм колебаний'
        '\n\nВыводит таблицу с жесткостями опор данного расчетного случая в левом нижнем углу каждого графика форм колебаний'
        '\n\nДопустимые значения:'
        '\n\tTrue - выводить таблицу жесткостей'
        '\n\tFalse - не выводить таблицу жесткостей'
        '\n\nЗначение по-умолчанию: True'
        '\n\n'
    ),
    placeholder='True',
    value='',
    datalist = [
        {'value':'True', 'description':'Выводить таблицу жесткостей'},
        {'value':'False', 'description':'Не выводить таблицу жесткостей'}
    ],
)

param11 = dict(
    feature = 'plot_dpi',
    description = (
        'Разрешение графиков в точках на дюйм'
        '\n\nУвеличение значения улучшает разрешение графиков, но при этом увеличивает размер файлов и время обработки. '
        'Рекомендуемый диапазон значений: от 100 до 200. Установленное ограничение: от 1 до 1000.'
        '\n\nФормат ввода: целое число.'
        '\n\nЗначение по-умолчанию: 100'
        '\n\n'
    ),
    placeholder='100',
    value='',
    datalist = [],
)

param12 = dict(
    feature = 'data_format',
    description = (
        'Формат сохранения таблиц'
        '\n\nДопустимые значения:'
        '\n\tcsv - Comma-Separated Values'
        '\n\txlsx - Excel Microsoft Office Open XML'
        '\n\nЗначение по-умолчанию: xlsx'
        '\n\nПримечание: При выборе формата csv также необходимо задать параметры data_encoding, data_sep, data_decimal. '
        'При выборе формата xlsx необходимо задать только параметр data_decimal.'
        '\n\n'
    ),
    placeholder='xlsx',
    value='',
    datalist = [
        {'value':'csv', 'description':'Comma-Separated Values'},
        {'value':'xlsx', 'description':'Excel Microsoft Office Open XML'},
    ],
)

param13 = dict(
    feature = 'data_encoding',
    description = (
        'Кодировка загружаемого / сохраняемого файла'
        '\n\nДопустимые значения:'
        '\n\tutf-8 - Unicode Transformation Format, 8-bit'
        '\n\tcp1251 - Windows-1251 (Russian)'
        '\n\nЗначение по-умолчанию: cp1251'
        '\n\nПримечание: Используется только для загрузки / сохранения файла в формате csv. '
        '\n\n'
    ),
    placeholder='cp1251',
    value='',
    datalist = [
        {'value':'utf-8', 'description':'Unicode Transformation Format, 8-bit'},
        {'value':'cp1251', 'description':'Windows-1251 (Russian)'},
    ],
)

param14 = dict(
    feature = 'data_sep',
    description = (
        'Разделитель ячеек загружаемого / сохраняемого файла'
        '\n\nПримеры наиболее часто используемых разделителей: ";", ",", "/t"'
        '\n\nЗначение по-умолчанию: ";"'
        '\n\nПримечание: Используется только для загрузки / сохранения файла в формате csv. '
        '\n\n'
    ),
    placeholder=';',
    value='',
    datalist = [
        {'value':';', 'description':'Точка с запятой'},
        {'value':',', 'description':'Запятая'},
    ],
)

param15 = dict(
    feature = 'data_decimal',
    description = (
        'Десятичный разделитель чисел с плавающей точкой загружаемого / сохраняемого файла'
        '\n\nДопустимые значения:'
        '\n\t"," - Десятичная запятая'
        '\n\t"." - Десятичная точка'
        '\n\nЗначение по-умолчанию: ","'
        '\n\nПримечание: При загрузке / сохранении файла в формате xlsx, используется только для строковых паременных. '
        '\n\n'
    ),
    placeholder=',',
    value='',
    datalist = [
        {'value':',', 'description':'Десятичная запятая'},
        {'value':'.', 'description':'Десятичная точка'},
    ],
)

param16 = dict(
    feature = 'frequency_units',
    description = (
        'Формат вывода частоты колебаний'
        '\n\nФормат частоты колебаний, используемый при выводе визуализации результатов'
        '\n\nДопустимые значения:'
        '\n\thertz - Герцы'
        '\n\tradian - Радианы в секунду'
        '\n\nЗначение по-умолчанию: hertz'
        '\n\n'
    ),
    placeholder='hertz',
    value='',
    datalist = [
        {'value':'hertz', 'description':'Гц'},
        {'value':'radian', 'description':'рад/с'},
    ],
)

param17 = dict(
    feature = 'horizontal_boundaries',
    description = (
        'Границы «горизонтальной» жесткости'
        '\n\nНижнее и верхнее значения «горизонтальной» жесткости, которые будут отображены на графике зависимости частот колебаний от жесткости. '
        '\n\nФормат ввода: Два числа с плавающей точкой или целых числа, введенных через запятую. '
        '\n\nЗначение по-умолчанию: 0.15, 0.30'
        '\n\nПримечание: Для отключения отображения необходимо ввести «None». '
        '\n\n'
    ),
    placeholder='0.15, 0.30',
    value='',
    datalist = [
        {'value':'None', 'description':'Отключить отображение'},
    ],
)

param18 = dict(
    feature = 'vertical_boundaries',
    description = (
        'Границы «вертикальной» жесткости'
        '\n\nНижнее и верхнее значения «вертикальной» жесткости, которые будут отображены на графике зависимости частот колебаний от жесткости. '
        '\n\nФормат ввода: Два числа с плавающей точкой или целых числа, введенных через запятую. '
        '\n\nЗначение по-умолчанию: 1.0, 1.5'
        '\n\nПримечание: Для отключения отображения необходимо ввести «None». '
        '\n\n'
    ),
    placeholder='1.0, 1.5',
    value='',
    datalist = [
        {'value':'None', 'description':'Отключить отображение'},
    ],
)

param19 = dict(
    feature = 'restricted_areas',
    description = (
        'Значение частот «запретных зон» в Гц'
        '\n\nСписок значений частот «запретных зон» в Гц, которые будут отображены на графике зависимости частот колебаний от жесткости. '
        '\n\nФормат ввода: числа с плавающей точкой или целые числа, введенные через запятую. '
        '\n\nЗначение по-умолчанию: 25.0, 50.0, 100.0'
        '\n\nПримечание: Для отключения отображения необходимо ввести «None». '
        '\n\n'
    ),
    placeholder='25.0, 50.0, 100.0',
    value='',
    datalist = [
        {'value':'None', 'description':'Отключить отображение'},
    ],
)

param20 = dict(
    feature = 'restricted_areas_width',
    description = (
        'Значение ширины окна «запретных зон» в процентах'
        '\n\nСписок значений ширины окна «запретных зон» в процентах для каждого значения частоты «запретных зон». '
        'Значение ширины окна «запретной зоны» соотносится со значением частоты по порядковому номеру записи. '
        '\n\nФормат ввода: числа с плавающей точкой или целые числа, введенные через запятую. '
        '\n\nЗначение по-умолчанию: 10.0, 10.0, 10.0'
        '\n\nПримечание: Количество значений ширины окна должно соотвествовать количеству значений частот «запретных зон». '
        'Если количество значений не совпадает, то для каждого значения частоты «запретной зоны» будет использовано первое указанное значение ширины окна. '
        'Соотвественно, допускается указывать одно значение ширины окна, которое будет использовано для всех укаханных значений частот «запретных зон». '
        '\n\n'
    ),
    placeholder='10.0, 10.0, 10.0',
    value='',
    datalist = [
        {'value':'None', 'description':'Отключить отображение'},
    ],
)


varibales_dict = dict(
    param1=param1,
    param2=param2,
    param3=param3,
    param4=param4,
    param5=param5,
    param6=param6,
    param7=param7,
    param8=param8,
    param9=param9,
    param10=param10,
    param11=param11,
    param12=param12,
    param13=param13,
    param14=param14,
    param15=param15,
    param16=param16,
    param17=param17,
    param18=param18,
    param19=param19,
    param20=param20,
)


@eel.expose
def get_varibales():
    get_init_params()
    return varibales_dict
