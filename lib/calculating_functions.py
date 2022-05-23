import os
from functools import wraps
from io import BytesIO
import base64

from typing import List, Dict, Tuple, Union

import pandas as pd
from pandas import read_csv, read_excel, DataFrame, MultiIndex
import numpy as np
import numpy.polynomial.polynomial as poly

from math import pi

from matplotlib import pyplot, style
style.use('ggplot')




class Rotor:

    def __init__(self):
        pass


    def load_rotor_data(
        self, 
        rotor_data_path, 
        data_format, 
        data_encoding, 
        data_sep, 
        data_decimal
    ):

        try:
            rotor_data = self.__load_data(
                data_path=rotor_data_path,
                format=data_format,
                encoding=data_encoding,
                sep=data_sep,
                decimal=data_decimal,
            )
        except Exception:
            return False

        status = self.__check_rotor_data_correct(rotor_data=rotor_data)

        if status == True:

            # Список столбцов жесткостей опор
            support_stiffness_list = [w for w in rotor_data.columns.to_list() if w.startswith('support_stiffness')]
            # Список наимнований столбцов жесткостей опор
            support_stiffness_names = [w.strip()[18:] for w in support_stiffness_list]

            # Первоначальные базовые столбцы
            base_columns = [w for w in rotor_data.columns if w not in support_stiffness_list]
            # Новые столбцы
            new_columns = sum([base_columns, support_stiffness_names],[])

            # Переименование столбцов жесткостей опор
            rotor_data.columns = new_columns

            # Выделение данных по опорам
            support_data = rotor_data[~rotor_data['support'].isna()]

            # Получение матрицы жесткостей
            stiffness_array = support_data[support_stiffness_names]
            stiffness_array.index.name = 'Участки опор'
            stiffness_array.columns.name = 'Варианты жёсткости опор'

            self.rotor_data = rotor_data
            self.support_data = support_data
            self.stiffness_array = stiffness_array

        return status


    def __check_rotor_data_correct(self, rotor_data:DataFrame):

        if (
            (rotor_data.shape[0] >= 2) and
            ('section' in rotor_data.columns.to_list()) and 
            ('length' in rotor_data.columns.to_list()) and 
            ('weight' in rotor_data.columns.to_list()) and
            ('stiffness' in rotor_data.columns.to_list()) and
            ('support' in rotor_data.columns.to_list()) and
            (len([w for w in rotor_data.columns.to_list() if w.startswith('support_stiffness')]) > 0) and
            (rotor_data.dtypes['section'] == np.int64) and
            (
                (rotor_data.dtypes['length'] == np.float64) or 
                (rotor_data.dtypes['length'] == np.int64)
            ) and 
            (
                (rotor_data.dtypes['weight'] == np.float64) or 
                (rotor_data.dtypes['weight'] == np.int64)
            ) and 
            (
                (rotor_data.dtypes['stiffness'] == np.float64) or 
                (rotor_data.dtypes['stiffness'] == np.int64)
            ) and
            (rotor_data[~rotor_data['support'].isna()][['support']].shape[0] > 0) and
            all([
                (rotor_data.dtypes[column] == np.float64) or 
                (rotor_data.dtypes[column] == np.int64) 
                for column in [w for w in rotor_data.columns.to_list() if w.startswith('support_stiffness')]
            ])
        ):

            status = True

        else:
            status = False

        return status


    def calculate_result_data(self, natural_frequencies_number, F1, FT1, fk, EPSF, normalize):

        try:
            # Получение собственных частот
            self.result, self.stiffness = self.__sob2014(
                section=self.rotor_data['section'].values,
                section_length=self.rotor_data['length'].values,
                section_weight=self.rotor_data['weight'].values,
                section_stiffness=self.rotor_data['stiffness'].values,
                support_sections=self.stiffness_array.index.values, 
                support_stiffness_names=self.stiffness_array.columns.values,
                stiffness_array=self.stiffness_array.values,
                natural_frequencies_number=natural_frequencies_number,
                F1=F1, FT1=FT1, fk=fk, EPSF=EPSF, normalize=normalize
            )
            status = True

        except Exception:
            status = False

        return status
    

    def __path_to_save_return(self, data_folder):

        # Папка сохранения результатов
        path_to_save = os.path.join(data_folder, 'result')

        # Проверка наличия рабочей папки в дирректории, в случае отсутствия папки - создает ее
        if not os.path.exists(path_to_save):
            try:
                os.mkdir(os.path.abspath(path_to_save))
            except Exception:
                raise  

        return path_to_save   


    def save_result_data(
        self, 
        data_folder:str,
        format:str,
        encoding:str,
        sep:str,
        decimal:str
    ):

        try:
            result = self.result.copy()
        except Exception:
            result = None

        if result is not None:

            status = True

            # Папка сохранения результатов
            path_to_save = self.__path_to_save_return(data_folder=data_folder)
            
            # Пути сохранения файлов
            path_to_result = os.path.join(path_to_save, f'result.{format}')

            # Замена булевых переменных в столбце индексов
            index = MultiIndex.from_tuples(
                tuples = [(w[0], w[1], (lambda x: x if x else '')(w[2])) for w in result.index],
                names = result.index.names
            )
            result.index = index

            # Экспорт данных
            if format == 'csv':
                result.to_csv(path_or_buf=path_to_result, sep=sep, decimal=decimal, encoding=encoding, index=True, header=True)
            elif format == 'xlsx':
                result.to_excel(excel_writer=path_to_result, index=True, header=True)
            else:
                status = False 

        else:
            status = False

        return status


    def save_stiffness_data(
        self, 
        data_folder:str,
        format:str,
        encoding:str,
        sep:str,
        decimal:str
    ):

        try:
            stiffness = self.stiffness.copy()
        except Exception:
            stiffness = None

        if stiffness is not None:

            status = True

            # Папка сохранения результатов
            path_to_save = self.__path_to_save_return(data_folder=data_folder)

            # Пути сохранения файлов
            path_to_stiffness = os.path.join(path_to_save, f'stiffness.{format}')
            
            # Экспорт данных
            if format == 'csv':
                stiffness.to_csv(path_or_buf=path_to_stiffness, sep=sep, decimal=decimal, encoding=encoding, index=True, header=True)   
            elif format == 'xlsx':
                stiffness.to_excel(excel_writer=path_to_stiffness, index=True, header=True)
            else:
                status = False
        
        else:
            status = False

        return status


    def prepare_visualizations1(
        self, 
        stiffness_on_plot, 
        plot_format, 
        plot_dpi, 
        frequency_units
    ):

        try:
            self.visualizations1 = self.__waveforms(
                result=self.result, 
                stiffness=self.stiffness, 
                stiffness_array=self.stiffness_array, 
                stiffness_on_plot=stiffness_on_plot, 
                plot_format=plot_format, 
                plot_dpi=plot_dpi,
                frequency_units=frequency_units
            )
            status = True
        except Exception:
            status = False

        return status


    def prepare_visualizations2(
        self, 
        plot_dpi,
        plot_format,  
        frequency_units, 
        horizontal_boundaries,
        vertical_boundaries,
        restricted_areas,
        restricted_areas_width,
    ):

        try:
            self.visualizations2 = self.__natural_frequencies_from_stiffness(
                stiffness=self.stiffness, 
                stiffness_array=self.stiffness_array, 
                plot_dpi=plot_dpi,
                plot_format=plot_format, 
                frequency_units=frequency_units,
                horizontal_boundaries=horizontal_boundaries,
                vertical_boundaries=vertical_boundaries,
                restricted_areas=restricted_areas,
                restricted_areas_width=restricted_areas_width,
            )
            status = True
        except Exception:
            status = False

        return status


    def save_visualizations1(self, data_folder):

        try:
            visualizations = self.visualizations1
        except Exception:
            visualizations = None

        status = self.__save_visualizations(
            data_folder=data_folder, 
            visualizations=visualizations
        )

        return status


    def save_visualizations2(self, data_folder):

        try:
            visualizations = self.visualizations2
        except Exception:
            visualizations = None

        status = self.__save_visualizations(
            data_folder=data_folder, 
            visualizations=visualizations
        )

        return status


    def __save_visualizations(self, data_folder, visualizations):

        if visualizations is not None:

            # Папка сохранения результатов
            path_to_save = self.__path_to_save_return(data_folder=data_folder)
            
            for i, visualization in enumerate(visualizations):

                figure = visualization.get('figure')
                name = visualization.get('name')
                format = visualization.get('format')
                dpi = visualization.get('dpi')

                # Наименование графика
                fname = os.path.join(path_to_save, f'plot_{name}.{format}')

                chart = figure.encode('utf-8')
                image = base64.b64decode(chart)
                
                with open(fname, "wb") as fh:
                    fh.write(image)

            status = True

        else:
            status = False

        return status


    def __load_data(
        self, 
        data_path:str,
        format:str,
        encoding:str,
        sep:str,
        decimal:str
    ):

        file = os.path.basename(data_path)
        file_name = ''.join(os.path.splitext(file)[:-1])
        file_format = ''.join(os.path.splitext(file)[-1])

        if file_format == '.csv':
            try:
                data = read_csv(data_path, sep=sep, engine='c', decimal=decimal, encoding=encoding)
            except FileNotFoundError:
                raise ValueError("Файл данных ротора по указанному пути не существует!")
            except UnicodeDecodeError:
                raise UnicodeError("Кодировка отдельных символов не поддерживается")
        elif file_format == '.xlsx':
            try:
                data = read_excel(data_path,  decimal=decimal, )
            except FileNotFoundError:
                raise ValueError("Файл данных ротора по указанному пути не существует!")
            except UnicodeDecodeError:
                raise UnicodeError("Кодировка отдельных символов не поддерживается")
        else:
            raise TypeError('Неверный формат файла')

        return data


    def __sob2014(
        self,
        section:List[int],
        section_length:List[float],
        section_weight:List[float],
        section_stiffness:List[float],
        support_sections:List[int],
        support_stiffness_names,
        stiffness_array,
        natural_frequencies_number:int, 
        F1:float=20.0,
        FT1:float=0.5,
        fk:float=1e4,
        EPSF:float=1e-7,
        normalize=True,
    ):
        """
        
        
        
        Parameters 
        ----------  
        section : List[int]
            Массив номеров участков
            
        section_length : List[float]
            Массив длин участков, м
            
        section_weight : List[float]
            Массив масс участков, кг
            
        section_stiffness : List[float]
            Массив жесткостей участков, (Н•м)
        
        support_sections : List[int]
            Массив номеров участков, которым соотвествуют опоры
            
        support_stiffness_names : List[str]
            Список наименований столбцов жесткостей
            
        stiffness_array : numpy.ndarray[float]
            Массив жесткостей опор валопровода (Н/м.*10**(-9)), где:
                columns - Наименования столбцов жесткостей опор (support_stiffness_names)
                index - Номера секций с опорами (support_sections)
            
        natural_frequencies_number : int
            Число собственных частот, подлежащих отысканию
        
        F1 : float, optional (default=20.0)
            Начальная частота диапзаона поиска собственных частот
        
        FT1 : float, optional (default=0.5)
            Шаг по частоте
        
        fk : float, optional (default=1e4)
            Верхняя граница диапазона поиска
        
        EPSF : float, optional (default=1e-7)
            Точность поиска частоты
        
        normalize : boolean, optional (default=True)
            Нормирование форм
        

        Note 
        ----------  

        section_count : int
            Число участков постоянной жесткости
            
        support_count : int
            Число опор
            
        KZ : numpy.ndarray[int]
            ??? Массив номеров участков, которым соотвествуют опоры
            
        length_from_start : numpy.ndarray[float]
            Массив длин участков от начала отсчета
            
        support_position : numpy.ndarray[float]
            Массив длин от начала отсчета для расположения опор (координаты опор)
            
        KW : int
            Число вариантов по жесткости опор

        """
        
        # Преобразование типа list к типу numpy.ndarray
        section = np.array(section)
        section_length = np.array(section_length)
        section_weight = np.array(section_weight)
        section_stiffness = np.array(section_stiffness)
        support_sections = np.array(support_sections)
        support_stiffness_names = np.array(support_stiffness_names)
        
        # Приведение переменных к стандартным размерностям
        section_stiffness = section_stiffness * 1e-9
        section_weight = section_weight * 1e-5
        
        # Определение числа участков постоянной жесткости
        section_count = section_length.shape[0]
        # Определение числа опор
        support_count = support_sections.shape[0]
        # Определение числа вариантов по жесткости опор
        KW = support_stiffness_names.shape[0]
        
        
        MZ = int((support_count / 2) + 2)
        KMZ = int(section_count / MZ)

        KZ = [i*KMZ-1 for i in range(1,MZ)]
        KZ.append(section_count-1)

        # Для совместимости шаг поиска уменьшается в 100 раз
        FT1 = FT1 / 100

        # Пересчет длин участков в длину от начала
        length_from_start = np.array([np.sum(section_length[:i+1]) for i in range(section_count)])
        # Заполнение массива длин от начала отсчета для расположения опор
        support_position = np.array([length_from_start[i] for i in support_sections])
        
        # Начальная частота диапозона поиска
        F1 = F1 / 100
        F11 = 100
        R1 = 3
                
        # Массив положений опор
        support = np.full(shape=(section_count,), fill_value=False)
        support[support_sections] = True
        
        
        summary_result_frame = np.array((), dtype=np.float64).reshape(section_count,0)
        summary_NP_frame = np.array((), dtype=np.int64)
        summary_IW_frame = np.array((), dtype=np.int64)
        summary_P_frame = np.array((), dtype=np.float64).reshape(natural_frequencies_number,0)
        
        # Проход по индексам вариантов по жесткости опор
        for IW in range(KW):
            
            result_frame, NP_frame, P_frame, IW_frame = self.__SOB(
                section_count=section_count,
                section_length=section_length,
                section_stiffness=section_stiffness,
                section_weight=section_weight,
                support_count=support_count,
                support_sections=support_sections,
                stiffness_array=stiffness_array,
                F1=F1, 
                IW=IW,
                MZ=MZ,
                FT1=FT1,
                fk=fk,
                KZ=KZ,
                EPSF=EPSF,
                F11=F11,
                R1=R1,
                natural_frequencies_number=natural_frequencies_number,
                normalize=normalize,
            )
            
            # Дозаполнение массива значением np.nan до размера natural_frequencies_number
            P_frame = np.append(
                P_frame, 
                np.full(shape=(natural_frequencies_number-P_frame.shape[0],), fill_value=np.nan, dtype=np.float64)
            )
            
            # Объединение локального массива с глобальным
            summary_result_frame = np.hstack((summary_result_frame, result_frame))    
            summary_NP_frame = np.hstack((summary_NP_frame, NP_frame))
            summary_P_frame = np.hstack((summary_P_frame, P_frame.reshape(-1,1)))
            summary_IW_frame = np.hstack((summary_IW_frame, IW_frame))
        


        result = DataFrame(summary_result_frame)
        result['section'] = section
        result['length_from_start'] = length_from_start
        result['support'] = support
        result = result.set_index(['section','length_from_start','support'])
        result.columns = MultiIndex.from_tuples(
            list(zip(support_stiffness_names[summary_IW_frame],summary_NP_frame)), 
            names=['Номер жесткости','Номер формы']
        )    
        
        stiffness = DataFrame(summary_P_frame)
        stiffness['Номер частоты'] = range(1,natural_frequencies_number+1)
        stiffness = stiffness.set_index(['Номер частоты'])
        stiffness.columns = support_stiffness_names
        stiffness.columns.name = 'Номер жесткости'
        
        return result, stiffness


    def __SOB(
        self,
        section_count, 
        section_length, 
        section_stiffness, 
        section_weight, 
        support_count, 
        support_sections,
        stiffness_array,
        F1, 
        IW, 
        MZ, 
        FT1, 
        fk, 
        KZ, 
        EPSF, 
        F11,
        R1, 
        natural_frequencies_number, 
        normalize,
    ):

        # Флаги
        flag_nn = True
        flag_new_KP = True
        flag_loop = True
        
        IM = 4 * MZ
        natural_frequency = 0
        IM1 = IM - 1
        IM2 = IM - 2
        IM3 = IM1 * IM1
        K = 0
        
        R5 = np.full(shape=(IM1,), fill_value=0.0, dtype=np.float64)
        RA = np.full(shape=(IM3,), fill_value=0.0, dtype=np.float64)
        RX = np.full(shape=(IM1+2,), fill_value=0.0, dtype=np.float64)
        
        
        # Глобальный результирующий массив
        result_frame = np.array((), dtype=np.float64).reshape(section_count,0)
        # Массив номеров частот
        NP_frame = list()
        # Массив жесткостей
        P_frame = list()
        # Массив индексов вариантов по жесткости опор
        IW_frame = list()
        
        
        while flag_loop:
            
            if flag_new_KP:
                KK = 1
                F = F1 # текущая частота равна начальной
                ft = FT1  
                flag_new_KP = False
            
            # Обнуление матрицы R
            R = np.full(shape=(IM,IM), fill_value=0.0, dtype=np.float64)

            K = K + 1
            J10 = 0

            # Если достигли верхней границы, то идём на выход
            if F > fk: 
                
                F1 = F
                flag_loop = False  
                
            else:

                # Единичная диагональная матрица 4х4
                A1 = np.eye(4, dtype=np.float64)
                # Нулевая матрица section_count х 4
                X = np.full(shape=(section_count,4), fill_value=0.0, dtype=np.float64)

                # Проход по индексам секций ротора
                for i in range(section_count):

                    # Длина i-го участка
                    P1 = section_length[i]
                    # Отношение длины i-го участка к жёсткости i-го участка
                    P2 = section_length[i] / section_stiffness[i]
                    # Произведение массы i-го участка и квдрата начальной частоты диапазона поиска
                    P3 = section_weight[i] * np.power(F,2)
                    P4 = P3 * P1
                    P5 = P4 * P2 / 2
                    P6 = P2 * P1 / 2
                    P7 = P6 * P1 / 3
                    P8 = 1 + P5 * P1 / 3 # Вероятнее всего, единица здесь не нужна

                    if flag_nn:

                        # Нулевая матрица 4х4
                        A2 = np.full(shape=(4,4), fill_value=0.0, dtype=np.float64)

                        for j in range(4):

                            a1 = P8 * A1[0,j] + P1 * A1[1,j] + P6 * A1[2,j] + P7 * A1[3,j]
                            a2 = P5 * A1[0,j] + A1[1,j] + P2 * A1[2,j] + P6 * A1[3,j]
                            a3 = P4 * A1[0,j] + A1[2,j] + P1 * A1[3,j]
                            a4 = P3 * A1[0,j] + A1[3,j]

                            A2[:,j] = np.array([a1, a2, a3, a4])

                        A1 = A2.copy()

                        if i == KZ[J10]:

                            M1 = J10 * 4

                            if J10 == 0:
                                M2 = 0
                                IT = 2
                                
                            else:
                                M2 = M1 - 2
                                IT = 4

                            for j in range(4):
                                for k in range(IT):
                                    R[M1+j,M2+k] = - A1[j,k] 

                            # Единичная диагональная матрица 4х4
                            A1 = np.eye(4, dtype=np.float64)

                            J10 += 1

                        for k in range(support_count):
                            if i == support_sections[k]:
                                for j in range(4):
                                    A1[3,j] = A1[3,j] - stiffness_array[k,IW] * A1[0,j] 


                    else:    

                        x1 = P8 * X1 + P1 * X2 + P6 * X3 + P7 * X4
                        x2 = P5 * X1 + X2 + P2 * X3 + P6 * X4
                        x3 = P4 * X1 + X3 + P1 * X4
                        x4 = P3 * X1 + X4 
                        
                        for j in range(support_count):
                            if i == support_sections[j]:
                                x4 -= stiffness_array[j,IW] * x1

                        X[i,:] = np.array([x1, x2, x3, x4])
                        
                        X1 = X[i,0]
                        X2 = X[i,1]
                        X3 = X[i,2]
                        X4 = X[i,3]


                        if i == KZ[J10]:

                            I1 = J10 * 4

                            X1 = RX[I1+1]
                            X2 = RX[I1+2]
                            X3 = RX[I1+3]
                            X4 = RX[I1+4]

                            J10 += 1    

                if flag_nn:

                    for i in range(IM2):
                        R[i,i+2] = 1

                    if ft <= 3 * EPSF:
                        for i in range(IM1):
                            if i <= 2:
                                R5[i] = - R[i+1,0]
                            for k in range(IM1):
                                RA[i+k*IM1] = R[i+1,k+1]  

                    if KK == 1:
                        
                        R, D1 = self.__OPRED(a=R, IPR=IM, JPR=8)
                        KK = 2
                        F += ft  
                        
                    else: 
                        
                        R, D2 = self.__OPRED(a=R, IPR=IM, JPR=8)
                        D3 = np.abs(D2) 

                        if ft >= EPSF:
                            
                            D = D1 * D2
                            
                            if D < 0:
                                F -= ft
                                ft /= R1
                                F += ft
                                
                            else:
                                D1 = D2
                                F += ft
                        else:

                            # Жесткость опоры
                            P = F*F11
                            # Сохранение параметра жесткости в глобальный массив
                            P_frame.append(P)

                            D1, RA = self.__MINV3(support_count=IM1, A_minv=RA)

                            for i in range(IM1):
                                D1 = 0
                                for j in range(3):
                                    D1 += RA[i+(j-0)*IM1] * R5[j]
                                RX[i] = D1

                            X1 = 1
                            X2 = RX[0]
                            X3 = 0
                            X4 = 0  
                            
                            F1 = F + 0.01
                            natural_frequency += 1
                            flag_nn = False

                else:
                    
                    if normalize:
                        # Нормирование форм
                        X[:,0] = X[:,0] / np.max(np.abs(X[:,0]))
                    
                    # Номер формы
                    NP_frame.append(natural_frequency)
                    # Сохранине индекса варианта по жесткости опор
                    IW_frame.append(IW)
                
                    # Объединение локального массива с глобальным
                    result_frame = np.hstack((result_frame, X[:,0].reshape(-1,1))) 


                    if natural_frequency < natural_frequencies_number:
                        flag_nn = True
                        flag_new_KP = True
                    else:
                        F1 = F
                        flag_loop = False  
                        
        return result_frame, np.array(NP_frame), np.array(P_frame), np.array(IW_frame)


    def __OPRED(self, a, IPR, JPR):
        """
        Ntemp : int
            Счетчик цикла в процедуре обращения матриц
        
        """
        
        DET = 1
        Ntemp = IPR - 1
        IBR = a.shape[0]

        for i in range(Ntemp):

            L2 = JPR
            
            if i + L2 > IPR: 
                L2 = IPR - i
                
            AM = np.abs(a[i,i])
            IM = i

            for lll in range(L2):
                if AM < np.abs(a[i+lll,i]):
                    IM = i + lll
                    AM = np.abs(a[IM,i])
                
            if IM != i: 
                for lll in range(IBR):
                    bbb = a[i,lll]
                    a[i,lll] = a[IM,lll]
                    a[IM,lll] = bbb
                DET = -DET
                
            bbb = a[i,i]
                
            for J in range(1,L2):
                J1 = i + J
                OT = a[J1,i] / a[i,i]
                for lll in range(i,IPR):
                    a[J1,lll] = a[J1, lll] - OT * a[i,lll]

        for i in range(IPR):
            J = IPR - i - 1
            if np.abs(DET) >= 1e30:
                DET /= 1e30
            DET *= a[J,J]

        return a, DET


    def __MINV3(self, support_count, A_minv):

        # Длина участка
        l_minv = np.full(shape=(support_count,), fill_value=0, dtype=np.int64)
        # Масса участка
        m_minv = np.full(shape=(support_count,), fill_value=0, dtype=np.int64)
        
        D = 1
        NK = -support_count
        K = 0
        
        while K < support_count:
            
            NK += support_count
            l_minv[K] = K
            m_minv[K] = K
            KK = NK + K
            BIGA = A_minv[KK]
            
            for J in range(K,support_count):
                iz = support_count * J
                for i in range(K,support_count):
                    ij = iz + i               
                    if (np.abs(BIGA) - np.abs(A_minv[ij])) < 0:
                        BIGA = A_minv[ij]
                        l_minv[K] = i
                        m_minv[K] = J
            
            J = l_minv[K]
            
            if (J - K) > 0:
                KI = K - support_count  
                for i in range(support_count):
                    KI += support_count
                    HOLD = -A_minv[KI]
                    JI = KI - K + J
                    A_minv[KI] = A_minv[JI]
                    A_minv[JI] = HOLD
            
            i = m_minv[K]
            
            if (i - K) > 0:
                JP = support_count * i
                for J in range(support_count):
                    JK = NK + J
                    JI = JP + J
                    HOLD = -A_minv[JK]
                    A_minv[JK] = A_minv[JI]
                    A_minv[JI] = HOLD
                
            if BIGA == 0:
                
                D = 0
                K = support_count
                
            else:       

                for i in range(support_count):
                    if (i - K) != 0:
                        IK = NK + i
                        A_minv[IK] = A_minv[IK] / (-BIGA)

                for i in range(support_count):
                    IK = NK + i
                    HOLD = A_minv[IK]
                    ij = i - support_count
                    for J in range(support_count):
                        ij += support_count
                        if (i - K) !=0 and (J - K) != 0:
                            KJ = ij - i + K
                            A_minv[ij] = HOLD * A_minv[KJ] + A_minv[ij]

                KJ = K - support_count
                
                for J in range(support_count):
                    KJ = KJ + support_count
                    if (J - K) != 0:
                        A_minv[KJ] = A_minv[KJ] / BIGA

                D *= BIGA
                A_minv[KK] = 1 / BIGA
                
                K += 1
        
        for K in range(support_count-1)[::-1]:

            i = l_minv[K]
            
            if (i - K) > 0:
                JQ = support_count * K
                JR = support_count * i
                for J in range(support_count):
                    JK = JQ + J
                    HOLD = A_minv[JK]
                    JI = JR + J
                    A_minv[JK] = -A_minv[JI]
                    A_minv[JI] = HOLD
                    
            J = m_minv[K]
            
            if (J - K) > 0:
                KI = K - support_count
                for i in range(support_count):
                    KI = KI + support_count
                    HOLD = A_minv[KI]
                    JI = KI - K + J
                    A_minv[KI] = -A_minv[JI]
                    A_minv[JI] = HOLD  

        return D, A_minv


    def __waveforms(
        self, 
        result, 
        stiffness, 
        stiffness_array, 
        stiffness_on_plot, 
        plot_format, 
        plot_dpi, 
        frequency_units
    ):
        """
        Visualization function
        
        
        Parameters 
        ----------  

        plot_format : str, optional (default='png')
            Формат сохранения графиков
            
        stiffness_on_plot : boolean, optional (default=True)
            Таблица жесткостей по опорам на графике

        plot_dpi : int, optional (default=100)
            Разрешение графиков в точках на дюйм
            
        """

        visualizations = list()

        # Массив длин участков от начала отсчета
        length_from_start = result.index.get_level_values('length_from_start').values
        # Массив наимнований столбцов жесткостей
        support_stiffness_names = result.columns.get_level_values('Номер жесткости').unique().values
        # Массив индексов положения опор
        support = result.index.get_level_values('support').values.astype(bool)
        # Массив длин от начала отсчета для расположения опор (координаты опор)
        support_position = length_from_start[support]

            
        # Перебор числа вариантов жёсткости опор
        for i, support_stiffness_name in enumerate(support_stiffness_names):
            

            natural_frequencies = result.iloc[
                :,result.columns.get_level_values('Номер жесткости') == support_stiffness_name
            ].columns.get_level_values('Номер формы').unique().values
            
            figure = self.__get_chart1(
                natural_frequencies=natural_frequencies, 
                support_stiffness_name=support_stiffness_name,
                length_from_start=length_from_start,
                support_position=support_position,
                result=result, 
                stiffness=stiffness, 
                stiffness_array=stiffness_array, 
                stiffness_on_plot=stiffness_on_plot, 
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                frequency_units=frequency_units,
            )

            visualizations.append({
                'figure':figure,
                'name':f'waveforms_{support_stiffness_name}',
                'format':plot_format,
                'dpi':plot_dpi,
            })

        return visualizations


    def __natural_frequencies_from_stiffness(
        self,
        stiffness,
        stiffness_array,
        plot_format,
        frequency_units,
        horizontal_boundaries,
        vertical_boundaries,
        restricted_areas,
        restricted_areas_width,
        plot_dpi,
    ):
        """
        Построение зависимости собственных частот колебаний от жесткости опор
        """

        visualizations = list()

        figure = self.__get_chart2(
            stiffness_array=stiffness_array, 
            stiffness=stiffness, 
            plot_format=plot_format,
            frequency_units=frequency_units, 
            horizontal_boundaries=horizontal_boundaries,
            vertical_boundaries=vertical_boundaries,
            restricted_areas=restricted_areas,
            restricted_areas_width=restricted_areas_width,
            plot_dpi=plot_dpi
        )

        visualizations.append({
            'figure':figure,
            'name':'natural_frequencies_from_stiffness',
            'format':plot_format,
            'dpi':plot_dpi,
        })

        return visualizations


    def __get_image(func):

        def image_buffering(figure, format, dpi):
            # Сохранение графика в буфер
            buffer = BytesIO()
            figure.savefig(buffer, format=format, bbox_inches='tight', dpi=dpi)
            buffer.seek(0)
            image = buffer.getvalue()
            # Очистка буфера
            buffer.close()
            # Закрытие графика
            pyplot.close()
            chart = base64.b64encode(image)
            chart = chart.decode('utf-8')
            return chart

        @wraps(func)
        def inner(*args, **kwargs):
            pyplot.switch_backend('AGG')
            # Turn the interactive mode off
            pyplot.ioff()
            # get figure
            result = func(*args, **kwargs)
            # get png (or other format) from figure
            chart = image_buffering(
                figure = result.get('figure'),  
                format=result.get('format'), 
                dpi = result.get('dpi'),
            )
            return chart
        return inner


    @__get_image
    def __get_chart1(
        self, 
        natural_frequencies, 
        support_stiffness_name,
        length_from_start,
        support_position,
        result, 
        stiffness, 
        stiffness_array, 
        stiffness_on_plot, 
        plot_format,
        plot_dpi,
        frequency_units: Union['radian','hertz']='hertz', 
    ):

        if (
            not isinstance(result, DataFrame) or 
            not isinstance(stiffness_array, DataFrame) or 
            not isinstance(stiffness, DataFrame)
        ):
            raise TypeError('Некорректный исходных формат данных!')
        
        if frequency_units not in ['radian', 'hertz']:
            raise ValueError('Некорректная единица частоты!')

        frequency_units_dict = {
            'radian':'рад/с', 
            'hertz':'Гц',
        }


        figure, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(16, 8), dpi=plot_dpi, sharex=False, sharey=False)
        
        # Перебор числа собственных частот
        for j, natural_frequency in enumerate(natural_frequencies):
            
            try:
                # Получение данных из массива
                y = result[(support_stiffness_name,natural_frequency)].values

                # Частота в рад/с
                frequency = stiffness.loc[natural_frequency,support_stiffness_name]

                if frequency_units == 'hertz':
                    # Частота в Гц
                    frequency = self.radian_to_hertz(frequency)

                # Построение кривой
                ax.plot(
                    length_from_start, 
                    y, 
                    label=f'{natural_frequency} ({frequency.round(2)} {frequency_units_dict.get(frequency_units)})'
                )
                
            except Exception:
                continue
            
        # Нанесение вертикальный линий, соотвествующих опорам
        for position in support_position:
            ax.axvline(position, color='g', linestyle='-', linewidth=1)        
            
        # Сохранение размеров осей    
        ax_shape = ax.axis()
        
        if stiffness_on_plot:
            stiffness_simple = stiffness_array.loc[:,[support_stiffness_name]]
            stiffness_simple = stiffness_simple.reset_index(level=0)
            stiffness_simple.columns = ['Опора','Жесткость']
            text = stiffness_simple.to_string(index=False)
            ax.text(
                0.02*(ax_shape[1]-ax_shape[0])+ax_shape[0], 
                0.02*(ax_shape[3]-ax_shape[2])+ax_shape[2], 
                text, 
                fontsize=8,
                bbox=dict(facecolor='red', alpha=0.5)
            ) 
            
        # Установка наименований графиков
        ax.set_title(label=f'Вариант жёсткости опор {support_stiffness_name}')
    
        # Установка параметров сетки
        ax.minorticks_on()
        ax.grid(which='major', axis='both', color = 'gray', linewidth='0.5', linestyle='-')
        ax.grid(which='minor', axis='both', color = 'gray', linewidth='0.5', linestyle=':')
                
        # Установка легенды
        ax.legend(loc='upper right')     
        
        #  Добавляем подписи к осям
        ax.set_xlabel('Координаты опор, м')
        ax.set_ylabel('')

        pyplot.tight_layout()

        return {'figure':figure, 'format':plot_format, 'dpi':plot_dpi}


    def hertz_to_radian(self, hertz:float) -> float:
        return hertz*2*pi


    def radian_to_hertz(self, radian:float) -> float:
        return radian/(2*pi)


    def __get_frequency_boundaries(
        self,
        x:np.ndarray,
        y:np.ndarray,
        horizontal_boundaries:Tuple[float,float]=None,
        vertical_boundaries:Tuple[float,float]=None,
    ) -> Dict[str,dict]:
        
        frequency_boundaries = dict()
        
        # Подготовка коэффициентов полинома
        coefs = poly.polyfit(x, y, x.shape[0])

        if horizontal_boundaries is not None:
        
            if horizontal_boundaries[0] in x:
                horizontal_frequency_bottom = y[x == horizontal_boundaries[0]][0]
            else:
                horizontal_frequency_bottom = poly.polyval(horizontal_boundaries[0], coefs)

            if horizontal_boundaries[-1] in x:
                horizontal_frequency_upper = y[x == horizontal_boundaries[-1]][0]
            else:
                horizontal_frequency_upper = poly.polyval(horizontal_boundaries[-1], coefs)
                
            frequency_boundaries['horizontal'] = {
                'frequency_bottom': horizontal_frequency_bottom,
                'frequency_upper': horizontal_frequency_upper,            
            }
            

        if vertical_boundaries is not None:
            
            if vertical_boundaries[0] in x:
                vertical_frequency_bottom = y[x == vertical_boundaries[0]][0]
            else:
                vertical_frequency_bottom = poly.polyval(vertical_boundaries[0], coefs)

            if vertical_boundaries[-1] in x:
                vertical_frequency_upper = y[x == vertical_boundaries[-1]][0]
            else:
                vertical_frequency_upper = poly.polyval(vertical_boundaries[-1], coefs)
                
            frequency_boundaries['vertical'] = {
                'frequency_bottom': vertical_frequency_bottom,
                'frequency_upper': vertical_frequency_upper,         
            }
            
        return frequency_boundaries


    @__get_image
    def __get_chart2(
        self,
        stiffness_array: DataFrame, 
        stiffness: DataFrame, 
        plot_format: str,
        frequency_units: Union['radian','hertz']='radian', 
        horizontal_boundaries: Tuple[float,float]=(0.15,0.30),
        vertical_boundaries: Tuple[float,float]=(1.0,1.5),
        restricted_areas: List[float]=None,
        restricted_areas_width: List[float]=None,
        plot_dpi=100
    ):
        
        if (
            not isinstance(stiffness_array, DataFrame) or 
            not isinstance(stiffness, DataFrame)
        ):
            raise TypeError('Некорректный исходных формат данных!')
        
        if frequency_units not in ['radian', 'hertz']:
            raise ValueError('Некорректная единица частоты!')
            
        frequency_units_dict = {
            'radian':'рад/с', 
            'hertz':'Гц',
        }
        
        colors = pyplot.get_cmap(name='Set1').colors
        colors_g = pyplot.get_cmap(name='tab20c').colors

        figure, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(12, 16), dpi=plot_dpi, sharex=False, sharey=False)
        
        left_ax = stiffness_array.iloc[0,-1]

        x = stiffness_array.mean(axis=0).values[::1]

        for i, natural_frequency in enumerate(stiffness.index.values):

            # Частота в рад/с
            y = stiffness.loc[natural_frequency,:].values[::1]
            
            if frequency_units == 'hertz':
                # Частота в Гц
                y = self.radian_to_hertz(y)
            
            # Построение кривой
            ax.plot(x, y, label=f'Собственная частота №{natural_frequency}', color=colors[i-1])

                
            frequency_boundaries_dict = self.__get_frequency_boundaries(
                x,y,
                horizontal_boundaries=horizontal_boundaries,
                vertical_boundaries=vertical_boundaries,
            )

            t = 0.15
            t1 = 0.10
            t2 = 0.02

            if horizontal_boundaries is not None:

                horizontal_frequency = frequency_boundaries_dict.get('horizontal')

                # Нижняя граница
                ax.hlines(
                    y=horizontal_frequency.get('frequency_bottom'), 
                    xmin=horizontal_boundaries[0],  xmax=horizontal_boundaries[-1]+t,
                    linestyle='dashed', linewidth=1,
                    color=colors[i-1]
                )
                ax.scatter(
                    x=horizontal_boundaries[0], 
                    y=horizontal_frequency.get('frequency_bottom'), 
                    color='black',
                    s=15
                )
                # Верхняя граница
                ax.hlines(
                    y=horizontal_frequency.get('frequency_upper'), 
                    xmin=horizontal_boundaries[-1],  xmax=horizontal_boundaries[-1]+t,
                    linestyle='dashed', linewidth=1,
                    color=colors[i-1]
                )
                ax.scatter(
                    x=horizontal_boundaries[-1], 
                    y=horizontal_frequency.get('frequency_upper'), 
                    color='black',
                    s=15
                )
                ax.vlines(
                    x=horizontal_boundaries[-1]+t1,
                    ymin=horizontal_frequency.get('frequency_bottom'),
                    ymax=horizontal_frequency.get('frequency_upper'),
                    linestyle='dashed', linewidth=1,
                    color=colors[i-1]
                )        
                ax.text(
                    x=horizontal_boundaries[-1]+t1+t2, 
                    y=(horizontal_frequency.get('frequency_bottom')+horizontal_frequency.get('frequency_upper'))/2-3, 
                    s=f'$\Delta P_{i+1}^г$',
                    rotation=0,
                    fontsize=12
                )


            if vertical_boundaries is not None:

                vertical_frequency = frequency_boundaries_dict.get('vertical')

                # Нижняя граница
                ax.hlines(
                    y=vertical_frequency.get('frequency_bottom'), 
                    xmin=vertical_boundaries[0],  xmax=vertical_boundaries[-1]+t,
                    linestyle='dashed', linewidth=1,
                    color=colors[i-1]
                )
                ax.scatter(
                    x=vertical_boundaries[0], 
                    y=vertical_frequency.get('frequency_bottom'), 
                    color='black',
                    s=15
                )
                # Верхняя граница
                ax.hlines(
                    y=vertical_frequency.get('frequency_upper'), 
                    xmin=vertical_boundaries[-1],  xmax=vertical_boundaries[-1]+t,
                    linestyle='dashed', linewidth=1,
                    color=colors[i-1]
                )
                ax.scatter(
                    x=vertical_boundaries[-1], 
                    y=vertical_frequency.get('frequency_upper'), 
                    color='black',
                    s=15
                )
                ax.vlines(
                    x=vertical_boundaries[-1]+t1,
                    ymin=vertical_frequency.get('frequency_bottom'),
                    ymax=vertical_frequency.get('frequency_upper'),
                    linestyle='dashed', linewidth=1,
                    color=colors[i-1]
                )     
                ax.text(
                    x=vertical_boundaries[-1]+t1+t2, 
                    y=(vertical_frequency.get('frequency_bottom')+vertical_frequency.get('frequency_upper'))/2-3, 
                    s=f'$\Delta P_{i+1}^в$',
                    rotation=0,
                    fontsize=12
                )


        if (restricted_areas is not None) and (restricted_areas_width is not None):

            if not len(restricted_areas_width) == len(restricted_areas):
                restricted_areas_width = [restricted_areas_width[0]] * len(restricted_areas)
            
            for j, (restricted_area, width) in enumerate(zip(restricted_areas,restricted_areas_width)):
                
                # Частота в Гц
                restricted_area_bottom = restricted_area * (1-(width/100)/2)
                restricted_area_center = restricted_area
                restricted_area_upper = restricted_area * (1+(width/100)/2)
                
                if frequency_units == 'radian':
                    # Частота в рад/с
                    restricted_area_bottom = self.hertz_to_radian(restricted_area_bottom)
                    restricted_area_center = self.hertz_to_radian(restricted_area_center) 
                    restricted_area_upper = self.hertz_to_radian(restricted_area_upper) 
                
                # Нижняя граница
                ax.hlines(
                    y=restricted_area_bottom, 
                    xmin=0,  xmax=left_ax,
                    linestyle='solid', linewidth=1,
                    color='darkgray'
                )
                # Верхняя граница
                ax.hlines(
                    y=restricted_area_upper, 
                    xmin=0,  xmax=left_ax,
                    linestyle='solid', linewidth=1,
                    color='darkgray'
                )   
                ax.fill_between(
                    x=[0,5], 
                    y1=[restricted_area_bottom]*2, 
                    y2=[restricted_area_upper]*2,
                    color=colors_g[-(j+1)],
                    alpha = 0.5,
                    label=f'Запретная зона {round(restricted_area_center,1)} {frequency_units_dict.get(frequency_units)} ({round(restricted_area_bottom,1)}-{round(restricted_area_upper,1)})'
                )
        
        
        # Установка параметров сетки
        ax.minorticks_on()
        ax.grid(which='major', axis='both', color = 'gray', linewidth='0.5', linestyle='-')
        ax.grid(which='minor', axis='both', color = 'gray', linewidth='0.5', linestyle=':')

        # Установка легенды
        ax.legend(loc='upper left')     

        #  Добавляем подписи к осям
        ax.set_xlabel('Жесткость опоры, Н/м')
        ax.set_ylabel(f'Частота, {frequency_units_dict.get(frequency_units)}')
        
        # Установка наименования графика
        ax.set_title(label=f'Зависимость собственных частот колебаний системы от жесткости опор')

        pyplot.tight_layout()

        return {'figure':figure, 'format':plot_format, 'dpi':plot_dpi}
