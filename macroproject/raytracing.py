# coding: utf-8
# license: GPLv3
import functools
import numpy as np
import os
import pandas as pd
import pvtrace as pvt
import trimesh as trm
import vtk


#класс модели трассировки лучей
class RayTraceModel:
    def __init__(self, modelname):
        #название модели
        self.modelname = modelname
        #сетки класса trimesh, задающие внешнюю и внутреннюю стенки тел (сосудов)
        self.meshes = []
        #массив показателей преломления тел (стенок и содержимого)
        self.RI = []
        #массив значений сдвигов тел по осям
        self.shift = None
        #массивы координат и направлений лучей
        self.pos = []
        self.dr = []
        #угол падения солнечных лучей относительно горизонтали
        self.angle = 0
        
    def loadMeshes(self, fnames, scale = 0.1):
        '''
        Функция загружает данные тел из файлов и генерирует соответствующие сетки
        fnames - имена файлов формата csv (разделитель ;), в котором построчно записаны 
        координаты точек: номер точки; координата x; координата y.
        scale - масштаб плоской фигуры
        Если все файлы существуют функция возвращает True, в противном случае
        массив сеток сбрасывается и возвращается False
        '''
        #сброс массива
        self.meshes = []
        self.RI = []
        for fname in fnames:
            #если файл существует
            if os.path.exists(fname):           
                self.meshes.append(self.trimeshRevolveBody(fname, scale, 0))
                self.RI.append(1.5)
            else:
                self.meshes = []
                self.RI = []
                self.shift = None
                return False
        self.shift = np.zeros((len(self.meshes), 3))
        return True
            
    def rotateMesh(self, mesh, angles = [0, 90, 0]):
        '''
        Функция осуществляет вращение сетки на заданные углы вокруг 
        осей x, y, z
        mesh - сетка trimesh
        angles - углы поворота относительно осей
        '''
        currmesh = mesh
        ang = np.pi * np.array(angles) / 180
        Mx = np.array([[1, 0, 0], 
                       [0, np.cos(ang[0]), - np.sin(ang[0])], 
                       [0, np.sin(ang[0]), np.cos(ang[0])]])
        My = np.array([[np.cos(ang[1]), 0, np.sin(ang[1])],
                       [0, 1, 0],
                       [-np.sin(ang[1]), 0, np.cos(ang[1])]])
        Mz = np.array([[np.cos(ang[2]), np.sin(ang[2]), 0],
                       [np.sin(ang[2]), np.cos(ang[2]), 0],
                       [0, 0, 1]])
        M = Mx * My * Mz
        currmesh.vertices = np.dot(currmesh.vertices, M)
        return currmesh
    
    def setRI(self, RI):
        '''
        Функция задает массив показателей преломления тел (стенок и содержимого)
        '''
        self.RI = RI
    
    def setShift(self, shift):
        '''
        Функция задает массив сдвигов тел (стенок и содержимого)
        '''        
        self.shift = shift
        
    def getMeshes(self):
        '''
        Функция возвращает массив сеток класса trimesh, 
        задающих внешнюю и внутреннюю стенки тел (сосудов)
        '''
        return self.meshes
        
    def traceRays(self, angle = 45, raysCol = 1000, showrenderer = False):
        '''
        Функция производит трассировку солнечных лучей, падающих под заданным углом,
        через тела, и возвращет координаты точек лучей 
        и соответствующих направлений, а также количество лучей на единицу площади
        angle - угол падения солнечных лучей к горизонтали
        raysCol - количество трассируемых лучей
        showrenderer - флаг визуализации трассировки встроенным просмотрщиком
        '''
        self.angle = angle
        #Определение границ первого сосуда по горизонталям и вертикали
        maxX = (self.meshes[0].vertices[:,0] + self.shift[0, 0]).max()
        maxZ = (self.meshes[0].vertices[:,2] + self.shift[0, 2]).max()
        minZ = (self.meshes[0].vertices[:,2] + self.shift[0, 2]).min()
        maxsize = max(2 * maxX, maxZ - minZ)
        # Создание расчетного пространства
        space = pvt.Node(
            name = 'Space',
            geometry = pvt.Sphere(
                radius = maxsize * 10,
                material = pvt.Material(refractive_index = 1.0),
            )
        )
        
        # Создание внешних стенок сосудов
        outer = []
        for i in range(0, len(self.meshes), 2):
            outer.append(pvt.Node(
                name = 'Outer (vessel) ' + str(i // 2),
                geometry = pvt.Mesh(self.meshes[i],
                material = pvt.Material(refractive_index = self.RI[i]),
                ),
                parent = space,
                location = (self.shift[i, 0], self.shift[i, 1], 
                            self.meshes[i].center_mass[2] + self.shift[i,2])
            ))
        # Создание внутренних стенок сосудов - внешних границ жидкости
        inner = []
        for i in range(1, len(self.meshes), 2):
            inner.append(pvt.Node(
                name = 'Inner (liquid) ' + str(i // 2),
                geometry = pvt.Mesh(self.meshes[i],
                material = pvt.Material(self.RI[i]),
                ),
                parent = outer[i // 2],
                location = (0, 0, self.meshes[i].center_mass[2])
            ))
       
        # Создание источника света
        light = pvt.Node(
            name = 'Light',
            light = pvt.Light(position = functools.partial(pvt.rectangular_mask, maxsize, 2 * maxX)),
            parent = space
        )
        # Поворот источника света на заданный угол
        light.rotate(np.pi * (1 / 2 + angle / 180), (0, 1, 0))
        light.location = (-maxsize * np.cos(np.pi * angle / 180), 0, 
                          maxsize * np.sin(np.pi * angle / 180))    
        
    
        
        # Трассировка лучей
        scene = pvt.Scene(space)
        if showrenderer:
            renderer = pvt.MeshcatRenderer(wireframe=True, open_browser=True)
            renderer.render(scene)
        
        positions = []
        directions = []
        for ray in scene.emit(raysCol):
            #пошаговая трассировка лучей
            steps = pvt.photon_tracer.follow(scene, ray,  emit_method = 'full')
            path, events = zip(*steps)
            #добавление к результирующим кортежам лучей, которые
            #прошли через сосуд
            if (len(path) > 2):
                pos, dr = self.parseRay(path)
                #удаление лучей, которые не направлены в сторону 'целевой' плоскости
                lastdr = dr[len(dr) - 1]
                if lastdr[0] >= 0:
                    positions.append(pos)
                    directions.append(dr)
                    if showrenderer:
                        renderer.add_ray_path(path)
        self.pos = positions
        self.dr = directions
        return positions, directions, raysCol / (8 * maxsize * maxX)
    
    
    def simulate(self, minAngle = 0, maxAngle = 90, raysCol = 1000):
        '''
        Функция производит расчет трассировки лучей через заданные объекты при 
        изменении угла их падения в заданном диапазоне
        minangle - нижняя граница угла падения солнечных лучей к горизонтали
        maxangle - верхняя граница угла падения солнечных лучей к горизонтали
        raysCol - количество трассируемых лучей
        Возвращает массив вида
        
        '''
        data = []
        #цикл перебора углов из заданного диапазона
        for currAngle in range(minAngle, maxAngle + 1):
            #трассировка
            currpos, currdir, raysdens = self.traceRays(currAngle, raysCol, 
                                                        showrenderer = False)
            #запись результатов в vtu-файлы, запись сеток объектов только для первого угла
            self.model2vtk(showRays = False, showMesh = False, 
                           writeMesh = True if currAngle == minAngle else False, 
                           writeRays = True)
            for cp in currpos:
                #формирование результатов моделирования: угол, входная 'плотность' лучей
                cdata = [currAngle, raysdens]
                #пройденный лучем путь через тела и координаты выходного участка луча
                rw = self.rayway(cp)
                cdata.append(rw[0][0])
                cdata.append(rw[0][2])
                cdata.append(rw[0][1])
                cdata.append(rw[1][0])
                cdata.append(rw[1][2])
                cdata.append(rw[1][1])
                cdata.append(rw[2])
                data.append(cdata)
                
        #преобразование в pandas формат и запись в csv-файл
        npData = np.array(data)
        pdData = pd.DataFrame(npData, columns = ['Angle', 
                                                 'RayDensity',
                                                 'X1', 'Y1', 'Z1',
                                                 'X2', 'Y2', 'Z2', 'RayWay'])
        pdData.to_csv(self.modelname + '\\trace_results.csv', sep=';')        
        return data
            
        
    def model2vtk(self, showRays = False, showMesh = False, writeMesh = False, writeRays = False):
        '''
        Функция генерирует визуализацию результатов моделирования в терминах vtk
        showRays - если True, то производится вывод лучей, в противном 
        случае не производится
        showMesh - если True, то производится вывод сеток объектов, в противном 
        случае не производится
        writeMesh - если True, то производится запись сеток объектов, в противном 
        случае не производится
        writeRays - если True, то производится запись лучей, в противном 
        случае не производится
        Запись результатов в формате vtu производится в папку modelname с именем вида:
        название сетки_угол падения солнечных лучей.vtu
        '''
        #Разделение сеток внешних и внутренних стенок объекта и генерация
        #соответствующих сеток в терминах vtk с учетом смещений
        outer = self.alltrimesh2vtk(self.meshes[0::2], self.shift[0::2, :])
        if len(self.meshes) > 1:
            inner = self.alltrimesh2vtk(self.meshes[1::2], self.shift[1::2, :]) 
        #Генерация в терминах vtk лучей
        if showRays or writeRays:
            rays = self.allray2vtk(self.pos)
        #Создание vtk рендерера, в который передаются все сгенерированные данные
        renderer = vtk.vtkRenderer()
        if showMesh:
            #создание визуализации внешних стенок объектов
            outMapper = vtk.vtkDataSetMapper()
            outMapper.SetInputData(outer)
            outActor = vtk.vtkActor()
            outActor.SetMapper(outMapper)
            #цвет - черный, непрозрачность - 0.5
            outActor.GetProperty().SetColor(0, 0, 0)
            outActor.GetProperty().SetEdgeColor(0, 0, 0)
            outActor.GetProperty().SetOpacity(0.5)
            outActor.GetProperty().EdgeVisibilityOn()
            renderer.AddActor(outActor)
            
            #создание визуализации внутреннего содержимого объектов
            inMapper = vtk.vtkDataSetMapper()
            inMapper.SetInputData(inner)
            inActor = vtk.vtkActor()
            inActor.SetMapper(inMapper)
            #цвет - синий, непрозрачность - 0.5
            inActor.GetProperty().SetColor(0, 0, 255)
            inActor.GetProperty().SetEdgeColor(0, 0, 0)
            inActor.GetProperty().SetOpacity(0.5)
            inActor.GetProperty().EdgeVisibilityOn()
            renderer.AddActor(inActor)  
        
        if showRays:
            #создание визуализации лучей
            rayMapper = vtk.vtkDataSetMapper()
            rayMapper.SetInputData(rays)
            rayActor = vtk.vtkActor()
            rayActor.SetMapper(rayMapper)
            #цвет - красный, непрозрачность - 1
            rayActor.GetProperty().SetColor(255, 0, 0)
            rayActor.GetProperty().SetEdgeColor(255, 0, 0)
            rayActor.GetProperty().SetOpacity(1)
            rayActor.GetProperty().EdgeVisibilityOn()
            renderer.AddActor(rayActor)
        
        #запись данных
        #если директории не существует, то она создается
        if not os.path.exists(self.modelname):
            os.mkdir(self.modelname)
        
        if writeMesh or writeRays:
            writer = vtk.vtkXMLUnstructuredGridWriter()
        if writeMesh:
            writer.SetInputDataObject(outer)
            writer.SetFileName(self.modelname + '\outer.vtu')
            writer.Write()
            
            if len(self.meshes) > 1:
                writer.SetInputDataObject(inner)
                writer.SetFileName(self.modelname + '\inner.vtu')
                writer.Write()    
        
        if writeRays:
            writer.SetInputDataObject(rays)
            writer.SetFileName(self.modelname + '\\rays' + str(self.angle) + '.vtu')
            writer.Write()         
     
        #фон
        renderer.SetBackground(255, 255, 255)
        #настройка камеры
        renderer.ResetCamera()
        renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        return renderer

    def trimeshRevolveBody(self, fname, scale = 0.1, shift = 0):
        '''
        Функция генерирует тело путем вращения плоской фигуры, заданной координатами
        точек, вокруг вертикальной оси.
        Параметры:
        fname - имя файла формата csv (разделитель ;), в котором построчно записаны 
        координаты точек: номер точки; координата x; координата y.
        scale - масштаб плоской фигуры
        shift - смещение плоской фигуры по вертикальной оси
        '''
        #загрузка данных из csv-файла
        data = pd.read_csv(fname, delimiter = ';')
        #названия столбцов
        names = data.columns.tolist()
        #считывание координат
        x = np.array(data[names[1]])
        y = np.array(data[names[2]])
        
        # масштабирование и смещение фигуры
        x = x * scale
        y = y * scale + shift
        
        #создание линий
        lines = [[x[i], y[i]] for i in range(len(x))]
        #формирование тела вращения
        return trm.creation.revolve(lines, sections = 360)
    
    
    def rayway(self, pos):
        '''
        Функция возвращает длину пути, который прошел луч, за исключением 
        участков вне тел (сосудов) и координаты последнего участка луча
        '''
        wayLen = 0
        for i in range(2, len(pos) - 1):
            dist = 0
            for j in range(3):
                dist += (pos[i][j] - pos[i - 1][j]) ** 2
            wayLen += dist ** 0.5
        return pos[len(pos) - 2], pos[len(pos) - 1], wayLen

    def raysways(self, pos):
        '''
        Функция возвращает длины путей, которые прошли лучи из массива, 
        за исключением участков вне тел (сосудов) и координаты последних 
        участков лучей
        '''
        res = []
        for p in pos:
            res.append(rayway(p))
        return res

    
    def trimesh2vtk(self, trimeshSource):
        '''
        Функция конвертирует сетку класса trimesh в сетку в терминах VTK
        trimeshSource - сетка класса trimesh
        '''
        #узлы сетки
        nodes = trimeshSource.vertices
        #треугольники сетки
        triangles = trimeshSource.faces
        # Сетка в терминах VTK
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        # Точки сетки в терминах VTK
        points = vtk.vtkPoints()
        # Обход всех точек сетки
        for i in range(0, len(nodes)):
            # Вставка новой точки в сетку VTK
            points.InsertNextPoint(nodes[i, 0], nodes[i, 2], nodes[i, 1])
        # Загрузка точек в сетку
        unstructuredGrid.SetPoints(points)
        # Вставка треугольников в сетку
        for i in range(0, len(triangles)):
            tri = vtk.vtkTriangle()
            for j in range(0, 3):
                tri.GetPointIds().SetId(j, triangles[i,j])
            unstructuredGrid.InsertNextCell(tri.GetCellType(), tri.GetPointIds())
        return unstructuredGrid
    
    def alltrimesh2vtk(self, trimeshSource, shift):
        '''
        Функция конвертирует сетку класса trimesh в сетку в терминах VTK
        trimeshSource - массив сеток класса trimesh
        shift - массив смещений сеток по трем осям
        '''
        #загрузка узлов и треугольников первой сетки
        nodes = trimeshSource[0].vertices + shift[0]
        triangles = trimeshSource[0].faces
        for i in range(1, len(trimeshSource)):
            #загрузка треугольников последующих сеток с учетом смещения индекса узлов
            triangles = np.concatenate([triangles, trimeshSource[i].faces + len(nodes)], axis = 0)
            #загрузка узлов последующих сеток
            nodes = np.concatenate([nodes, trimeshSource[i].vertices + shift[i]], axis = 0)
            
        # Сетка в терминах VTK
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        # Точки сетки в терминах VTK
        points = vtk.vtkPoints()
        # Обход всех точек сетки
        for i in range(0, len(nodes)):
            # Вставка новой точки в сетку VTK
            points.InsertNextPoint(nodes[i, 0], nodes[i, 2], nodes[i, 1])
        # Загрузка точек в сетку
        unstructuredGrid.SetPoints(points)
        # Вставка треугольников в сетку
        for i in range(0, len(triangles)):
            tri = vtk.vtkTriangle()
            for j in range(0, 3):
                tri.GetPointIds().SetId(j, triangles[i, j])
            unstructuredGrid.InsertNextCell(tri.GetCellType(), tri.GetPointIds())
        return unstructuredGrid
    
    def ray2vtk(self, pos):
        '''
        Функция конвертирует результаты трассировки луча в сетку в терминах VTK
        '''
        # Сетка в терминах VTK
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        # Точки сетки в терминах VTK
        points = vtk.vtkPoints()
        pcount = 0
        for i in range(0, len(pos)):
            # Вставка новой точки в сетку VTK
            coords = pos[i]
            points.InsertPoint(pcount, [coords[0], coords[2], coords[1]])
            # Вставка сегмента луча
            if i > 0:
                unstructuredGrid.InsertNextCell(vtk.VTK_LINE, 2, [pcount - 1, pcount])
            else:
                unstructuredGrid.InsertNextCell(vtk.VTK_VERTEX, 1, [pcount])
            pcount += 1
        # Загрузка точек в сетку
        unstructuredGrid.SetPoints(points)
        return unstructuredGrid   
    
    def allray2vtk(self, pos):
        '''
        Функция конвертирует результаты трассировки множества лучей 
        в сетку в терминах VTK
        '''
        # Сетка в терминах VTK
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        # Точки сетки в терминах VTK
        points = vtk.vtkPoints()
        pcount = 0
        for currpos in pos:
            for i in range(0, len(currpos)):
                # Вставка новой точки в сетку VTK
                coords = currpos[i]
                points.InsertPoint(pcount, [coords[0], coords[2], coords[1]])
                # Вставка сегмента луча
                if i > 0:
                    unstructuredGrid.InsertNextCell(vtk.VTK_LINE, 2, [pcount - 1, pcount])
                else:
                    unstructuredGrid.InsertNextCell(vtk.VTK_VERTEX, 1, [pcount])
                pcount += 1
        # Загрузка точек в сетку
        unstructuredGrid.SetPoints(points)
        return unstructuredGrid

    def parseRayStr(self, path):
        '''
        Функция выделяет из результатов трассировки и возвращает координаты точек 
        преломления луча pos и соответствующие векторы направлений dir
        path - результаты трассировки луча
        '''
        #считывание результатов
        s = path
        #удаление пробелов
        s = s.replace(' ', '')
        #создание массивов
        pos = []
        dr = []
        #поиск координат начальной точки луча
        p = s.find('pos=')
        #цикл разбивки трассировки
        while p > -1:
            #считывание координат
            s = s[p + 5:]
            p = s.find(')')
            pos.append(list(map(float, s[:p].split(','))))
            s = s[p + 1:]
            #поиск и считывание направления
            p = s.find('dir=')
            s = s[p+5:]
            p = s.find(')')
            dr.append(list(map(float, s[:p].split(','))))
            #поиск следующих координат луча
            p = s.find('pos=')
        return pos, dr

    def parseRay(self, path):
        '''
        Функция выделяет из результатов трассировки и возвращает координаты точек 
        преломления луча pos и соответствующие векторы направлений dir
        path - результаты трассировки луча
        '''
        #создание массивов
        pos = []
        dr = []
        for i in range(len(path)):
            pos.append(path[i].position)
            dr.append(path[i].direction)
        return pos, dr
