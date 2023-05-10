# coding: utf-8
# license: GPLv3
import numpy as np
import pandas as pd
import raytracing as rt
import lightspots as ls
import postprocess as pp

'''
модуль для расчета термического воздействия солнечного света, сфокусированного
трехлитровой банкой с водой
'''

if __name__ == '__main__':
    
    #название модели
    modelname = 'Jar3L'
    #создание модели для трассировки
    test =  rt.RayTraceModel(modelname)
    #формирование сеток стенок банки и воды
    test.loadMeshes(['jar3l_out.csv', 'jar3l_in.csv'], scale = 0.1)
    #показатели преломления
    test.setRI([1.5, 1.33])
    #сдвиги сеток
    shift = np.array([[0, 0, 0], [0, 0, 0]])
    test.setShift(shift)
    
    #предварительный просмотр сеток и трассировки лучей
    #pos, dr, x = test.traceRays(angle = 70, raysCol = 10, showrenderer = False) 
    
    #renderer = test.model2vtk(showMesh = False, showRays = True, writeMesh = False, writeRays = False)
    
    #расчет трассировки в заданном диапазоне углов.
    test.simulate(minAngle = 0, maxAngle = 70, raysCol = 5000)
    
    #создание модели для поиска пятен сфокусированного света
    lfm = ls.LightSpotModel(modelname)
    #загрузка результатов трассировки
    lfm.loaddata()
    #грубый поиск пятен в заданной области
    lfm.roughfindAllSpots(minAngle = 0, maxAngle = 70, astep = 1, absF = 0.001,
                          xmin = 0, xmax = 15, ymin = 5, ymax = -12, 
                          zmin = 0, zmax = 0, size = 1, step = 1, minrays = 40)
    # точный поиск пятен 
    lfm.accuratefindAllSpots(fname = 'Rough_results.csv',
                             absF = 0.001, srange = [0.5, 0.5, 0.0],
                             size = 0.2, step = 0.5, minrays = 50)
    #постпроцессинг результатов
    ppr = pp.PostProcResults(modelname)
    #загрузка результатов
    ppr.loaddata("FinalSpot_results.csv")
    #расчет температуры для черной бумаги
    temp, maxtemp = ppr.findTemperature(mu = 0.95, hc = 9.5, T0 = 22 + 273, e = 0.9)
    #расчет температуры для белой бумаги
    #temp, maxtemp = ppr.findTemperature(mu = 0.4, hc = 9.5, T0 = 22 + 273, e = 0.9)    
    #построение графиков
    ppr.plotgraphs()
    #загрузка данных из csv-файла
    data = pd.read_csv('jar3l_out.csv', delimiter = ';')
    ves = np.array([np.array(data["x"]), np.array(data["y"]) + 108]).transpose()
    ppr.plotXY(vessel = ves)