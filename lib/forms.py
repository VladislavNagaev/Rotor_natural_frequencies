import os
from typing import Literal, Optional, Union, List
from pydantic import BaseModel, Field, ValidationError, validator


class Variables(BaseModel):
    param1: Optional[str] = Field(alias='data_folder')
    param2: Optional[str] = Field(alias='rotor_data_path')
    param3: Optional[int] = Field(alias='natural_frequencies_number', ge=1, le=10)
    param4: Optional[float] = Field(alias='F1', ge=0.0)
    param5: Optional[float] = Field(alias='FT1', ge=0.0)
    param6: Optional[float] = Field(alias='fk', ge=0.0)
    param7: Optional[float] = Field(alias='EPSF', ge=0.0)
    param8: Optional[bool] = Field(alias='normalize')
    param9: Optional[Literal[
        'ps','eps','pdf','pgf', 'png', 'raw', 'rgba', 'svg', 'svgz', 'jpg', 'jpeg', 'tif', 'tiff'
    ]] = Field(alias='plot_format')
    param10: Optional[bool] = Field(alias='stiffness_on_plot')
    param11: Optional[int] = Field(alias='plot_dpi', ge=1, le=1000)
    param12: Optional[Literal[
        'csv','xlsx',
    ]] = Field(alias='data_format')
    param13: Optional[Literal[
        'utf-8','cp1251',
    ]] = Field(alias='data_encoding')
    param14: Optional[str] = Field(alias='data_sep')
    param15: Optional[Literal[',','.']] = Field(alias='data_decimal')
    param16: Optional[Literal['hertz','radian']] = Field(alias='frequency_units')
    param17: Optional[str] = Field(alias='horizontal_boundaries')
    param18: Optional[str] = Field(alias='vertical_boundaries')
    param19: Optional[str] = Field(alias='restricted_areas')
    param20: Optional[str] = Field(alias='restricted_areas_width')

    @validator('param1')
    def data_folder_checker(cls, data_folder:str) -> str:
        # Проверка существования рабочей папки
        if not os.path.exists(data_folder):
            raise ValidationError("Directory not exist!")
        return data_folder

    @validator('param2')
    def rotor_data_path_checker(cls, rotor_data_path:str) -> str:

        # Проверка существования файла
        if not os.path.exists(rotor_data_path):
            raise ValidationError("File not exist!")

        # Проверка расширения файла
        filename, file_extension = os.path.splitext(rotor_data_path)
        if not file_extension.lower() == '.csv' and not file_extension.lower() == '.xlsx':
            raise ValidationError("File extension not supported!")

        return rotor_data_path
    
    @validator('param17','param18')
    def boundaries_checker(cls, boundaries:str) -> Union[List[float],None]:
        
        if (boundaries == None) or (str(boundaries).lower() == 'none'):
            boundaries = None
        else:
            boundaries_list = boundaries.strip().split(',')
            boundaries = list()
            
            if not len(boundaries_list) == 2:
                raise ValidationError("")
            
            for boundary in boundaries_list:
                try:
                    boundary = float(boundary.strip())
                except ValueError as e:
                    raise ValidationError("")
                boundaries.append(boundary)
                
            for boundary in boundaries:
                if not (0.0 <= boundary <= 10.0):
                    raise ValidationError("")
                
            if boundaries[0] >= boundaries[-1]:
                raise ValidationError("")
                
        return boundaries

    @validator('param19')
    def restricted_areas_checker(cls, areas:str) -> Union[List[float],None]:
        
        if (areas == None) or (str(areas).lower() == 'none'):
            areas = None
        else:
            areas_list = areas.strip().split(',')
            areas = list()
            
            if not len(areas_list) >= 1:
                raise ValidationError("")
            
            for area in areas_list:
                try:
                    area = float(area.strip())
                except ValueError as e:
                    raise ValidationError("")
                areas.append(area)
                
            for area in areas:
                if not (0.0 <= area <= 10000.0):
                    raise ValidationError("")
                
        return areas

    @validator('param20')
    def restricted_areas_width_checker(cls, areas_width:str) -> Union[List[float],None]:
        
        if (areas_width == None) or (str(areas_width).lower() == 'none'):
            areas_width = None
        else:
            areas_width_list = areas_width.strip().split(',')
            
            areas_width = list()
            
            if not len(areas_width_list) >= 1:
                raise ValidationError("")
            
            for area_width in areas_width_list:
                try:
                    area_width = float(area_width.strip())
                except ValueError as e:
                    raise ValidationError("")
                areas_width.append(area_width)
                
            for area_width in areas_width:
                if not (0.0 < area_width <= 100.0):
                    raise ValidationError("")
                
        return areas_width



