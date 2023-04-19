from firedrake import *

n = 10
mesh = UnitSquareMesh(n)

degree = 1
V = VectorFunctionSpace(mesh, "RT", degree, dim=2)
Q = VectorFunctionSpace(mesh, "DG", degree-1, dim=2)
T = VectorFunctionSpace(mesh, "HDivTrace", degree-1, dim=2)
eta = Constant(1.0)

W = V * Q * T * T

u, p, uhat, phat = TrialFunctions(W)
v, q, vhat, qhat = TestFunctions(W)

a = (
    inner(v, perp(u))*dx - div(v)*p*dx 
    + jump(v)*phat*dS - inner(dot(v,n), dot(u,n))*ds
    - inner(dot(v('+'), n('+')), dot(u('+'), n('+'))
    )
