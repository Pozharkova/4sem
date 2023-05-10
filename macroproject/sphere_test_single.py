# coding: utf-8
# license: GPLv3
import numpy as np
import raytracing as rt
import lightspots as ls
import trimesh as trm
import postprocess as pp

'''модуль для валидации и верификации моделей на примере прохождения лучей 
солнечного света через стеклянный шар
'''

if __name__ == '__main__':
    
    modelname = 'Sphere_single'
    #Создание модели для трассировки
    test =  rt.RayTraceModel(modelname)
    #Создание сплошной стеклянной сферы радиусом 75 мм
    #грубый вариант сетки
    #test.meshes = [trm.creation.uv_sphere(radius = 7.5)]
    #точный вариант сетки
    test.meshes = [trm.creation.uv_sphere(radius = 7.5, count = [360, 360])]
    #Показатель преломления
    test.setRI([1.53])
    #Сдвиг сферы
    shift = np.array([[0, 0, 0]])
    test.setShift(shift)
    #Трассировка лучей (в количестве 1000 - быстрый расчет) для оценки фокусного расстояния
    #test.simulate(minAngle = 0, maxAngle = 70, raysCol = 1000)  
    
    #Трассировка лучей (в количестве 40000) для последующей оценки интенсивности
    #в ограниченном диапазоне углов
    test.simulate(minAngle = 0, maxAngle = 0, raysCol = 40000)
    
    #создание модели для расчета интенсивности сфокусированного потока
    lfm = ls.LightSpotModel(modelname)
    #загрузка данных
    lfm.loaddata()
    #грубый поиск пятна с максимальной интенсивностью в окрестностях фокуса 
    #с показателем поглощения для стелка 0.015
    lfm.roughfindAllSpots(minAngle = 0, maxAngle = 0, astep = 1, absF = 0.015,
                          xmin = 9.5, xmax = 11.5, ymin = -1, ymax = 1, 
                          zmin = 0, zmax = 0, size = 0.5, step = 1, minrays = 20)
    #точный поиск пятна с максимальной интенсивностью. 
    #Площадь пятна соответствует исходной плотности сгенерированных лучей'''
    lfm.accuratefindAllSpots(fname = 'Rough_results0_0.csv',
                             absF = 0.015, srange = [1.5, 0.0, 0.0],
                             size = 0.1, step = 1, minrays = 100)
    ppr = pp.PostProcResults(modelname)
    #загрузка результатов
    ppr.loaddata("FinalSpot_results.csv")
    #расчет температуры
    temp, maxtemp = ppr.findTemperature(mu = 0.7, hc = 7.0, T0 = 22 + 273, e = 0.8)
    #построение графиков
    ppr.plotgraphs(xshift = 0, yshift = 0, smooth = False)
    x = np.array([75 * np.cos(i * np.pi / 180) for i in range(360)])
    y = np.array([75 * np.sin(i * np.pi / 180) for i in range(360)])
    ves = np.array([x, y]).transpose()
    ppr.plotXY(xshift = 0, yshift = 0, vessel = ves) 