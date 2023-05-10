# coding: utf-8
# license: GPLv3
import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy.signal import savgol_filter
import os
import logging
logging.getLogger("matplotlib").setLevel(logging.CRITICAL + 1)

class PostProcResults:
    def __init__(self, modelname):
        #название модели
        self.modelname = modelname
        #массивы с координатами точек с максимальной интенсивностью
        self.p = None
        #массив углов
        self.angle = None
        #массивы c "относительной плотностью потока" и температурой
        self.dens = None
        self.temp = None
    
    def loaddata(self, fname):
        """
        Функция загружает результаты расчета из файла
        """
        # Загрузка данных и разбивка по соответствующим массивам
        data = pd.read_csv(self.modelname+ "\\" + fname, delimiter = ";")
        names = data.columns.tolist()
        self.angle = np.array(data["Angle"])
        self.dens = np.array(data["MaxFlowDens"])
        self.p = np.array([np.array(data["X"]), np.array(data["Y"]), np.array(data["Z"])]).transpose()

    def equation(self, T, initFD, mu = 0.85, hc = 5.0, T0 = 22 + 273, e = 0.85):
        '''
        Функция возвращает значение уравнения для расчета температуры
        T - текущее значение температуры, К
        initFD - начальные плотности потока солнечного излучения для различных углов 
        mu - коэффициент, учитывающий отражение поверхности, которая нагревается светом
        hc - коэффициент конвективной теплоотдачи поверхности, которая нагревается светом
        T0 - температура окружающего воздуха, К
        e - коэффициент излучения поверхности, которая нагревается светом
        '''
        s = 5.67 * 10 ** -7
        return initFD * mu - 2 * (hc * (T - T0) + e * s * (T ** 4 - T0 ** 4))
    
    def findTemperature(self, initFD = None, 
                        mu = 0.85, hc = 5.0, T0 = 22 + 273, e = 0.85):

        """
        Функция производит расчет температуры в области пятна сфокусированного света, 
        определяет максимальное значение и записывает результаты в файл
        initFD - начальные плотности потока солнечного излучения для различных углов 
        mu - коэффициент, учитывающий отражение поверхности, которая нагревается светом
        hc - коэффициент конвективной теплоотдачи поверхности, которая нагревается светом
        T0 - температура окружающего воздуха, К
        e - коэффициент излучения поверхности, которая нагревается светом
        """
        #Если плотность потока не задана, то исполльзуются значения аппроксимированные на основе
        #соответствующей таблицы
        if not initFD:
            initFD = [540 * (1 - np.exp(-0.04 * self.angle[i])) + 400 for i in range(len(self.angle))]
        temp = []
        for i in range(len(self.angle)):
            #плотность потока в области пятна
            Q = self.dens[i] * initFD[i]
            #начальная температура
            T = T0
            #прямой поиск решения с шагом 1 К
            preveq = self.equation(T, Q, mu, hc, T0, e)
            T += 1
            nexteq = self.equation(T, Q, mu, hc, T0, e)
            while nexteq > 0:
                prevq = nexteq
                T += 1
                nexteq = self.equation(T, Q, mu, hc, T0, e)
            if preveq > abs(nexteq):
                temp.append(T - 1)
            else:
                temp.append(T)
        self.temp = np.array(temp)
        #максимальная температура
        maxtemp = self.temp.max()
        #заполнение массивов и запись результатов расчета температуры в файл
        npData = np.array([self.angle, self.temp, self.p[:, 0], self.p[:, 1], self.p[:, 2]]).transpose()
        pdData = pd.DataFrame(npData, columns = ["Angle", 
                                                 "Temperature", "X", "Y", "Z"])
        pdData.to_csv(self.modelname + "\\Temp_results.csv", sep=";")         
        return self.temp, maxtemp

    def plotgraphs(self, xshift = -65, yshift = 100, step = 1, smooth = True, Tk = 273 + 250):
        '''
        Функция строит графики по результатам расчета температур
        x - смещение значений х, мм (стенка сосуда)
        y - смещение значений по y, мм (дно сосуда)
        step - шаг между значениями выводимых на графики
        smooth - если True, то выводятся также сглаженные графики
        Tk - критическая температура для исследуемого материала
        '''
        fig, ax = pyplot.subplots(num = None, figsize = (10, 6))
        ax.scatter(self.angle[::step], self.temp[::step], c = "r", label = "Temperature")
        ax.scatter(self.angle[::step], self.p[::step, 0] * 10 + xshift, c = "b", label = "X")
        ax.scatter(self.angle[::step], self.p[::step, 1] * 10 + yshift, c = "g", label = "Y")
        ax.plot([self.angle[0], self.angle[-1]], [Tk, Tk], c = "orange", label = "Critical temperature")
        #сглаживание методом Савицкого-Голея
        if smooth:
            Temp = savgol_filter(self.temp[::step], 10, 3)
            x = savgol_filter(self.p[::step, 0], 10, 3)
            y = savgol_filter(self.p[::step, 1], 10, 3)
            ax.plot(self.angle[::step], Temp, c = "r")
            ax.plot(self.angle[::step], x * 10 + xshift, c = "b")
            ax.plot(self.angle[::step], y * 10 + yshift, c = "g")        
        ax.legend(loc = 7)
        ax.set_xlabel("Angle, " + r"$^{\circ}$")
        ax.set_ylabel("Temperature, K \nX, Y, mm")
        pyplot.grid(visible = True)
        pyplot.show()        

    def plotXY(self, xshift = 0, yshift = 100, step = 1, vessel = None, Tk = 273 + 250):
        '''
        Функция выводит координаты точек с максимальной температурой
        и проекцию сосуда
        x - смещение значений х, мм
        y - смещение значений по y, мм
        step - шаг между значениями выводимых на графики
        vessel - координаты точек проекции сосуда
        '''
        fig, ax = pyplot.subplots(num = None, figsize = (10, 6))
        ax.scatter(self.p[self.temp[::step] < Tk, 0] * 10 + xshift, 
                   self.p[self.temp[::step] < Tk, 1] * 10 + yshift, 
                   c = "green", 
                   label = "Max temperature points (" + r"$<T_k$" + ")")
        ax.scatter(self.p[self.temp[::step] >= Tk, 0] * 10 + xshift, 
                   self.p[self.temp[::step] >= Tk, 1] * 10 + yshift, 
                   c = "red", 
                   label = "Max temperature points (" + r"$\geq T_k$" + ")")        
        ax.plot(vessel[:, 0], vessel[:, 1], c = "b", label = "Vessel")
        ax.legend(loc = 1)
        ax.set_xlabel("X, mm")
        ax.set_ylabel("Y, mm")
        pyplot.grid(visible = True)
        pyplot.show()        


                
            