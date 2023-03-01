from fenics import *
from mshr import *

T = 2.5 
num_steps = 2500   
dt = T / num_steps 
mu = 0.001         
rho = 1            

channel = Rectangle(Point(0, 0), Point(1, 0.4))
triangle = Polygon([Point(0.15, 0.2), Point(0.23, 0.14), Point(0.23, 0.26)])
rhombus = Polygon([Point(0.35, 0.25), Point(0.45, 0.18), Point(0.55, 0.25), Point(0.45, 0.32)])
domain = channel - triangle - rhombus
mesh = generate_mesh(domain, 64)

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 1)'
walls    = 'near(x[1], 0) || near(x[1], 0.4)'
triangle = 'on_boundary && x[0] > 0.14 && x[0] < 0.24 && x[1] > 0.13 && x[1] < 0.27'
rhombus = 'on_boundary && x[0] > 0.34 && x[0] < 0.56 && x[1] > 0.17 && x[1] < 0.26'

inflow_profile = ('0.8', '0')

bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree = 2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_triangle = DirichletBC(V, Constant((0, 0)), triangle)
bcu_rhombus = DirichletBC(V, Constant((0, 0)), rhombus)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_triangle, bcu_rhombus]
bcp = [bcp_outflow]

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

U  = 0.5 * (u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))

F1 = rho * dot((u - u_n) / k, v) * dx \
   + rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx \
   + inner(sigma(U, p_n), epsilon(v)) * dx \
   + dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds \
   - dot(f, v) * dx
a1 = lhs(F1)
L1 = rhs(F1)

a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (1 / k) * div(u_) * q * dx

a3 = dot(u, v) * dx
L3 = dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

file_u = File('navier_stokes_figures/velocity.pvd')
file_p = File('navier_stokes_figures/pressure.pvd')

t = 0
for n in range(num_steps):

    
    t += dt

    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'petsc_amg')

    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'petsc_amg')

    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    file_u << u_
    file_p << p_

    u_n.assign(u_)
    p_n.assign(p_)

    print("Current time: %f / %f" % (t, T))
