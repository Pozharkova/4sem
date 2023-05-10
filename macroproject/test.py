# coding: utf-8
# license: GPLv3
import sys
import vtk
import numpy as np
import raytracing as rt
import lightspot as ls

if __name__ == "__main__":
    
    modelname = "Test"
    
    test =  rt.RayTraceModel(modelname)
    test.loadMeshes(["jar3l_out.csv", "jar3l_in.csv"], scale = 0.1)
    test.setRI([1.5, 1.33, 1.5, 1.33])
    shift = np.array([[0, 0, 0], [0, 0, 0]])
    test.setShift(shift)
    pos, dr, x = test.traceRays(angle = 70, raysCol = 10, showrenderer = False) 
    
    renderer = test.model2vtk(showMesh = False, showRays = True, writeMesh = False, writeRays = False)
    test.simulate(minAngle = 0, maxAngle = 1)
    
    lfm = ls.LightSpotModel(modelname)
    lfm.loaddata()
    angles, maxrays, maxspots = lfm.findAllSpots(minAngle = 0, maxAngle = 1, 
                                                 astep = 1, absF = 0.001,
                                                 xmin = 7, xmax = 20, 
                                                 ymin = -5, ymax = 5, 
                                                 zmin = -3, zmax = 3, 
                                                 size = 0.5, step = 1, 
                                                 minrays = 10)   
