"""Программа для моделирования падения твердого тела в жидкость
"""
import numpy as np

# PySPH
from pysph.base.utils import (get_particle_array_wcsph,
                              get_particle_array_rigid_body)
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, SummationDensity)
from pysph.sph.wc.basic import TaitEOSHGCorrection, MomentumEquation
from pysph.solver.application import Application
from pysph.sph.rigid_body import (
    BodyForce, SummationDensityBoundary, RigidBodyCollision, RigidBodyMoments,
    RigidBodyMotion, AkinciRigidFluidCoupling, RK2StepRigidBody)


# основные параметры модели
#масштаб в метрах
scale = 1e-3
#высота жидости
water_height = 150
#высота сосуда
tank_height = 400
#ширина жидкости равна ширине сосуда
water_width = 300
#высота на которой находится геометрический центр тела
figure_height = 400
#угол поворота тела относительно его геометрического центра
angle = 0
#имя файла с изображением тела
f_name = 'fopf.png'
#плотность жидкости
fluid_rho = 1000
#плотность тела
figure_rho = 833

def create_boundary(dx = 2, width = water_width, height = tank_height):
    '''
    Функция создает сосуд и масштабирует его размеры:
    dx - шаг между точками
    width, height - исходная (до масштабирования) ширина и высота сосуда
    '''
    #дно сосуда
    xb = np.arange(-2 * dx, width + 2 * dx, dx)
    yb = np.arange(-2 * dx, 0, dx)
    xb, yb = np.meshgrid(xb, yb)
    xb = xb.ravel()
    yb = yb.ravel()
    #стенки сосуда
    xl = np.arange(-2 * dx, 0, dx)
    yl = np.arange(0, height, dx)
    xl, yl = np.meshgrid(xl, yl)
    xl = xl.ravel()
    yl = yl.ravel()

    xr = np.arange(width, width + 2 * dx, dx)
    yr = np.arange(0, height, dx)
    xr, yr = np.meshgrid(xr, yr)
    xr = xr.ravel()
    yr = yr.ravel()
    #слияние массивов дна и стенок
    x = np.concatenate([xl, xb, xr])
    y = np.concatenate([yl, yb, yr])
    #масштабирование
    return x * scale, y * scale


def create_fluid(dx = 2, width = water_width, height = water_height):
    '''
    Функция создает "прямоугольник жидкости" и масштабирует его:
    dx - шаг между точками
    width, height - исходная (до масштабирования) ширина и высота прямоугольника
    '''
    xf = np.arange(0, width, dx)
    yf = np.arange(0, height, dx)
    xf, yf = np.meshgrid(xf, yf)
    xf = xf.ravel()
    yf = yf.ravel()
    #масштабирование
    return xf * scale, yf * scale


def create_figure(dx = 1, shiftx = water_width / 2, shifty = figure_height, r_angle = np.pi * angle / 180, name = f_name):
    '''
    Функция создает тело по изображению из файла и масштабирует его:
    dx - шаг между точками
    shiftx, shifty - положение по осям х и у геометрического центра тела
    r_angle - угол поворота тела относительно геометрического центра
    name - имя файла с изображением тела
    '''
    #загрузка изображения в виде массива точек с цветами в формате RGB
    from PIL import Image
    im = np.array(Image.open(name))
    #определение исходной ширины и высоты изображения
    width = im.shape[1]
    height = im.shape[0]
    #цвет "фона" - точек, которые не относятся к телу
    Col = np.array([255, 255, 255])
    #создание массива координат точек тела, цвет которых отличается от цвета "фона"
    x = []
    y = []
    for i in range(width):
        for j in range(height):
            if np.all(im[j, i] != Col):
                x.append(i * dx)
                y.append(height - j * dx)
    #центрирование тела
    x = np.array(x)
    x = x - (min(x) + max(x)) / 2
    y = np.array(y)
    y = y - (min(y) + max(y)) / 2
    #поворот тела на заданный угол и перемещение в заданную точку
    xr = x * np.cos(r_angle) - y * np.sin(r_angle) + shiftx
    yr = x * np.sin(r_angle) + y * np.cos(r_angle) + shifty
    #масштабирование тела
    return xr * scale, yr * scale

def get_density(y):
    '''
    функция плотности жидкости
    '''
    height = water_height
    c_0 = 2 * np.sqrt(2 * 9.81 * height * scale)
    rho_0 = 1000
    height_water_clmn = height * scale
    gamma = 7.
    _tmp = gamma / (rho_0 * c_0 ** 2)

    rho = np.zeros_like(y)
    for i in range(len(rho)):
        p_i = rho_0 * 9.81 * (height_water_clmn - y[i])
        rho[i] = rho_0 * (1 + p_i * _tmp) ** (1. / gamma)
    return rho


def model_preview():
    '''
    Функция выводит на экран исходные изображения сосуда, жидкости и тела
    '''
    import matplotlib.pyplot as plt
    #создание сосуда
    x_tank, y_tank = create_boundary()
    #создание жидкости
    x_fluid, y_fluid = create_fluid()
    #создание тела (кота)
    x_cat, y_cat = create_figure()
    #построение изображений в виде точек в координатной плоскости
    plt.scatter(x_fluid, y_fluid, s = 0.5)
    plt.scatter(x_tank, y_tank, s = 0.5)
    plt.scatter(x_cat, y_cat, s = 0.5)
    plt.show()

#класс приложения для моделирования взаимодействия твердого тела и жидкости
class RigidFluidCoupling(Application):
    def initialize(self):
        '''
        Инициализация класса
        '''
        #исходные параметры: размеры частиц, плотность жидкости и твердого тела
        #масса частиц и т.д.
        self.dx = 2 * scale
        self.hdx = 1.2
        self.rho = fluid_rho
        self.solid_rho = figure_rho
        #self.m = self.rho * self.dx * self.dx
        self.co = 2 * np.sqrt(2 * 9.81 * water_height * scale)
        self.alpha = 0.1

    def create_particles(self):
        """Функция создает частицы жидкости, сосуда и тела"""
        #генерация массивов координат частиц жидкости
        xf, yf = create_fluid()
        #определение параметров частиц жидкости
        m = self.rho * self.dx * self.dx
        rho = self.rho
        h = self.hdx * self.dx
        #создание жидкости в виде массива частиц с параметрами метода
        #WCSPH - Weakly Compressible SPH
        fluid = get_particle_array_wcsph(x = xf, y = yf, h = h, m = m, rho = rho,
                                         name = "fluid")
        #генерация массивов координат частиц сосуда
        xt, yt = create_boundary()
        #определение параметров частиц сосуда
        m = self.rho * self.dx * self.dx
        rho = self.rho
        rad_s = 2 / 2. * scale
        h = self.hdx * self.dx
        V = self.dx * self.dx
        #создание сосуда в виде массива частиц с параметрами метода WCSPH
        tank = get_particle_array_wcsph(x = xt, y = yt, h = h, m = m, rho = rho,
                                        rad_s = rad_s, V = V, name = "tank")
        for name in ['fx', 'fy', 'fz']:
            tank.add_property(name)

        #генерация массивов координат частиц тела
        dx = 1
        xc, yc = create_figure()
        #определение параметров частиц сосуда
        m = self.solid_rho * (dx * scale) ** 2
        rho = self.solid_rho
        h = self.hdx * self.dx
        rad_s = dx / 2. * scale
        V = dx * dx * scale * scale
        cs = 0.0
        #создание твердого тела (кота)
        cat = get_particle_array_rigid_body(x = xc, y = yc, h = h, m = m, rho = rho,
                                             rad_s = rad_s, V = V, cs = cs,
                                             name = "cat")

        return [fluid, tank, cat]

    def create_solver(self):
        '''
        Создание солвера
        '''
        #двумерная модель
        kernel = CubicSpline(dim = 2)
        #Интегрирование по методу EPEC - Evaluate-Predict-Evaluate-Correct
        integrator = EPECIntegrator(fluid = WCSPHStep(), tank = WCSPHStep(),
                                    cat = RK2StepRigidBody())
        #шаг по времени
        dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 2.

        print("Time step: %s" % dt)
        #Полное время моделирования
        tf = 2
        #создание солвера
        solver = Solver(
            kernel = kernel, dim = 2, integrator = integrator, dt = dt, tf = tf,
            adaptive_timestep = False)

        return solver

    def create_equations(self):
        '''
        Создание уравнений модели
        '''
        equations = [
            Group(equations = [
                BodyForce(dest = 'cat', sources = None, gy = -9.81),
            ], real=False),
            Group(equations = [
                SummationDensity(
                    dest = 'fluid',
                    sources = ['fluid'], ),
                SummationDensityBoundary(
                    dest = 'fluid', sources = ['tank', 'cat'], fluid_rho = self.rho)
            ]),

            # Уравнение состояния Тэйта
            Group(equations = [
                TaitEOSHGCorrection(dest = 'fluid', sources = None, rho0 = self.rho,
                                    c0 = self.co, gamma = 7.0)], real = False),
            Group(equations = [
                MomentumEquation(dest = 'fluid', sources = ['fluid'],
                                 alpha = self.alpha, beta = 0.0, c0 = self.co,
                                 gy = -9.81),
                AkinciRigidFluidCoupling(dest = 'fluid',
                                         sources = ['cat', 'tank']),
                XSPHCorrection(dest = 'fluid', sources = ['fluid', 'tank']),
            ]),
            Group(equations = [
                RigidBodyCollision(dest = 'cat', sources = ['tank'], kn = 1e5)
            ]),
            Group(equations = [RigidBodyMoments(dest = 'cat', sources = None)]),
            Group(equations = [RigidBodyMotion(dest = 'cat', sources = None)]),
        ]
        return equations

#запуск расчета
if __name__ == '__main__':
    #model_preview()
    app = RigidFluidCoupling()
    app.run()
