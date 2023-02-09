import gmsh
import math
import sys

gmsh.initialize()
gmsh.model.add("tor")

R = 5
r1 = 0.3
r2 = 1
lc = (r2 - r1) / 2

#правая окружность
x1 = [R, R + r2, R, R - r2, R]
z1 = [0, 0, r2, 0, -r2]

x2 = [R, R + r1, R, R - r1, R]
z2 = [0, 0, r1, 0, -r1]

p1 = []
p2 = []
for i in range(len(x2)):
    p1.append(gmsh.model.geo.addPoint(x2[i], 0, z2[i], lc))
    p2.append(gmsh.model.geo.addPoint(x1[i], 0, z1[i], lc))


a1 = []
a2 = []
for i in range(len(x2) - 1):
    a1.append(gmsh.model.geo.addCircleArc(p1[i + 1], p1[0], p1[i + 2] if i + 2 < len(x2) else p1[1]))
    a2.append(gmsh.model.geo.addCircleArc(p2[i + 1], p2[0], p2[i + 2] if i + 2 < len(x2) else p2[1]))

c1 = gmsh.model.geo.addCurveLoop(a1)
c2 = gmsh.model.geo.addCurveLoop(a2)
s1 = gmsh.model.geo.addPlaneSurface([c1, -c2])
gmsh.model.geo.synchronize()

x1 = [-R, -R + r1, -R, -R - r1, -R]
z1 = [0, 0, r1, 0, -r1]


x2 = [-R, -R + r2, -R, -R - r2, -R]
z2 = [0, 0, r2, 0, -r2]


p1 = []
p2 = []
for i in range(len(x2)):
    p1.append(gmsh.model.geo.addPoint(x2[i], 0, z2[i], lc))
    p2.append(gmsh.model.geo.addPoint(x1[i], 0, z1[i], lc))


a1 = []
a2 = []
for i in range(len(x2) - 1):
    a1.append(gmsh.model.geo.addCircleArc(p1[i + 1], p1[0], p1[i + 2] if i + 2 < len(x2) else p1[1]))
    a2.append(gmsh.model.geo.addCircleArc(p2[i + 1], p2[0], p2[i + 2] if i + 2 < len(x2) else p2[1]))

c1 = gmsh.model.geo.addCurveLoop(a1)
c2 = gmsh.model.geo.addCurveLoop(a2)
s2 = gmsh.model.geo.addPlaneSurface([c1, -c2])
gmsh.model.geo.synchronize()

gmsh.model.geo.revolve([(2, s1)], 0, 0, 0, 0, 0, 1, math.pi)
gmsh.model.geo.revolve([(2, s2)], 0, 0, 0, 0, 0, 1, math.pi/2)
gmsh.model.geo.revolve([(2, s1)], 0, 0, 0, 0, 0, 1, -math.pi/2)

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(3)

gmsh.write("tor.msh")
gmsh.write("tor.geo_unrolled")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
