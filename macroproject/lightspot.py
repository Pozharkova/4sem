# coding: utf-8
# license: GPLv3
import numpy as np
import pandas as pd
import vtk
import os


class LightSpotModel:
    def __init__(self, modelname):
        #название модели
        self.modelname = modelname
        #массивы с координатами точек, через которые проходят лучи
        self.p1 = None
        self.p2 = None
        #массив углов
        self.angle = None
        #массивы длин путей лучей и их начальной "плотности"
        self.rayway = None
        self.raydens = None 

        
    def loaddata(self):
        '''
        Функция загружает результаты трассировки из файла
        '''
        # Загрузка данных и разбивка по соответствующим массивам
        data = pd.read_csv(self.modelname+ "\\trace_results.csv", delimiter = ";")
        names = data.columns.tolist()
        self.angle = np.array(data["Angle"])
        self.raydens = np.array(data["RayDensity"])
        self.rayway = np.array(data["RayWay"])
        self.p1 = np.array([np.array(data["X1"]), np.array(data["Y1"]), np.array(data["Z1"])]).transpose()
        self.p2 = np.array([np.array(data["X2"]), np.array(data["Y2"]), np.array(data["Z2"])]).transpose()        
    
    def p2angle(self, p, angle):
        '''
        Функция делает выделяет из массива данные соответствующие заданному углу:
        p - исходный массив
        angle - заданный угол
        '''
        return p[np.array(self.angle == angle)]
        
    def point2line(self, point, p1, p2):
        '''
        Функция вычисляет расстояние от заданной точки до луча
        point - заданная точка
        p1, p2 - точки, через которые проходит луч
        '''
        v = p1 - point
        k = p2 - p1
        s = np.cross(v, k)
        return np.linalg.norm(s) / np.linalg.norm(k)
    
   
    def absortion (self, way, absF = 0.001):
        '''
        Функция вычисляет относительное снижение интенсивности света за счет
        поглощения
        way - расстояние, пройденное светом
        absF - коэффициент поглощения
        '''
        return 10 ** (-absF * way)
    
    def findSpots(self, p1, p2, rayway, raydens, absF = 0.001, 
                  xmin = 0, xmax = 20, ymin = - 15, ymax = 15, 
                  zmin = -5, zmax = 5, size = 0.5, step = 1, minrays = 10):
        '''
        Функция производит поиск "пятен", в которые сфокусированы лучи
        p1, p2 - массивы координат точек, через которые проходят лучи
        rayway - массив пройденных путей
        raydens - массив исходных "плотностей" лучей
        absF - коэффициент поглощения
        xmin, xmax, ymin, ymax, zmin, zmax - координаты параллепипеда, 
        ограничивающего область поиска
        size - радиус "пятна"
        step - шаг (в радиусах "пятна") между соседними "ячейками" поиска
        minrays - минимальная кратность превышения "плотности" лучей в "пятне" 
        отсекающего области фокусировки по сравнению с исходной
        Функция возвращает массив координат найденных "пятен" с соответствующими 
        "плотностями" лучей с учетом поглощения, а также максимальную "плотность"
        и координаты соответствующего пятна
        '''
        # максимальное значение выходной "плотности" и координаты пятна
        maxray = 0
        maxspot = [0, 0, 0]
        #массив координат "пятен" и "плотностей"
        spots = []
        #площадь пятна
        s = np.pi * size * size
        #перебор по всей области поиска с заданным шагом
        x = xmin
        while x <= xmax:            
            y = ymin
            while y <= ymax:
                z = zmin
                while z <= zmax:
                    
                    rayscount = 0
                    point = np.array([x, y, z])
                    for i in range(len(p1)):
                        # вычисление расстояния луча от центра пятна
                        dist = self.point2line(point, p1[i], p2[i])
                        # если луч попал в границы пятна, то
                        if  dist <= size:
                            # добавление относительной "интенсивности" с учетом поглощения и начального значения
                            rayscount += self.absortion(rayway[i], absF) / (s * raydens [i])  
                    # если пороговое значение превышено, то пятно добавляется к результатам
                    if rayscount >= minrays:
                        spots.append([rayscount, x, y, z])
                    # максимальное значение
                    if rayscount > maxray:
                        maxray = rayscount
                        maxspot = [x, y, z]
                    z += size * step
                y += size * step
            x += size * step
        return spots, maxray, maxspot    
    
    def findAllSpots(self, minAngle = 0, maxAngle = 90, astep = 1, absF = 0.001,
                      xmin = 0, xmax = 20, ymin = - 15, ymax = 15, 
                      zmin = -5, zmax = 5, size = 0.5, step = 1, minrays = 10):
        '''
        Функция производит поиск "пятен", в которые сфокусированы лучи, при
        заданном диапазоне углов падения солнечного света
        minAngle, maxAngle, astep - границы диапазона углов и шаг перебора 
        absF - коэффициент поглощения
        xmin, xmax, ymin, ymax, zmin, zmax - координаты параллепипеда, 
        ограничивающего область поиска
        size - радиус "пятна"
        step - шаг (в радиусах "пятна") между соседними "ячейками" поиска
        minrays - минимальная кратность превышения "плотности" лучей в "пятне" 
        отсекающего области фокусировки по сравнению с исходной
        Функция на каждом этапе прозводит запись результатов в формате vtk 
        и возвращает и записывает csv файл массив углов и соответствующие 
        максимальную "плотность" и координаты соответствующего пятна.
        '''
        # сброс результирующих массивов
        angles = []
        maxrays = []
        maxspots = []
        #цикл перебора углов
        for angle in range(minAngle, maxAngle + 1, astep):
            angles.append(angle)
            # срезы исходных данных для заданного угла
            p1 = self.p2angle(self.p1, angle)
            p2 = self.p2angle(self.p2, angle)
            rw = self.p2angle(self.rayway, angle)
            rd = self.p2angle(self.raydens, angle)
            # поиск пятен для текущего угла
            spot, maxray, maxspot = self.findSpots(p1, p2, rw, rd, absF, 
                                                   xmin, xmax, ymin, ymax, 
                                                   zmin, zmax, size, step, 
                                                   minrays)
            # обновление результирующих массивов
            maxspots.append(maxspot)
            maxrays.append(maxray)
            # запись текущих результатов в терминах vtk
            self.results2vtk(angle, spot)
        # запись итоговых результатов в csv-файл
        maxrays = np.array(maxrays)
        maxspots = np.array(maxspots)
        angles = np.array(angles)
        npData = np.array([angles, maxrays, maxspots[:, 0], maxspots[:, 1], 
                           maxspots[:, 2]]).transpose()
        pdData = pd.DataFrame(npData, columns = ["Angle", 
                                                 "MaxRays",
                                                 "X", "Y", "Z"])
        pdData.to_csv(self.modelname + "\\Spot_results" + str(minAngle) + "_" + str(maxAngle) + ".csv", sep=";") 
        return angles, maxrays, maxspots 
    

    def spot2vtk(self, spot):
        '''
        Функция конвертирует результаты поиска "пятен с повышенной интенсивностью" 
        в сетку в терминах VTK
        spot - массив пятен и "относительных интенсивностей"
        '''

        # Сетка в терминах VTK
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        # Точки сетки в терминах VTK
        points = vtk.vtkPoints()
        
        # Поле "относительных интенсивностей" лучей на точках сетки
        dens = vtk.vtkDoubleArray()
        dens.SetName("Flow density")
         
        # Вставка точек и соответствующих значений поля в сетку VTK
        for i in range(0, len(spot)):
            points.InsertNextPoint(spot[i][1], spot[i][2], spot[i][3])
            dens.InsertNextValue(spot[i][0])
            vert = vtk.vtkVertex()
            vert.GetPointIds().SetId(0, i)
            unstructuredGrid.InsertNextCell(vert.GetCellType(), vert.GetPointIds())
        # Загрузка точек и поля в сетку
        unstructuredGrid.SetPoints(points)
        unstructuredGrid.GetPointData().AddArray(dens)
        return unstructuredGrid
    
    def results2vtk(self, angle, spot):
        '''
        Функция записывает результаты в формате vtu в папку modelname с именем вида:
        spots_угол падения солнечных лучей.vtu
        angle - угол
        spot - массив с результатами
        '''
        #запись данных
        #если директории не существует, то она создается
        if not os.path.exists(self.modelname):
            os.mkdir(self.modelname)
        
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputDataObject(self.spot2vtk(spot))
        writer.SetFileName(self.modelname + "\spot_" + str(angle) + ".vtu")
        writer.Write()
            
   
    





