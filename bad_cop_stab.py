from firedrake import *

n = 10
mesh = UnitSquareMesh(n, n)

degree = 1
V = VectorFunctionSpace(mesh, "RT", degree, dim=2)
Q = VectorFunctionSpace(mesh, "DG", degree-1, dim=2)
T = VectorFunctionSpace(mesh, "HDiv Trace", degree-1, dim=2)
eta = Constant(1.0)
kappa = Constant(1.0)

f = as_vector([Constant(1.0), Constant(0.0)])

W = V * Q * T * T

u, p, uhat, phat = TrialFunctions(W)
v, q, vhat, qhat = TestFunctions(W)

J = as_tensor([[0, -1], [0, 1]])

def both(u):
    return u('+') + u('-')

n = FacetNormal(mesh)

a = (
    inner(v, kappa*dot(u, J))*dx - inner(div(v), p)*dx 
    + inner(both(inner(v, n)), phat)*dS
    - inner(dot(v,n), dot(u,n))*ds
    - inner(dot(v('+'), n('+')), dot(u('+'), n('+')) - uhat)/eta*dS
    - inner(dot(v('-'), n('-')), dot(u('-'), n('-')) + uhat)/eta*dS
    - inner(dot(v, n), dot(u, n) - uhat)/eta*ds
    + inner(q, dot(p, J) +  div(u))*dx
    + inner(qhat, both(inner(u, n)))*dS - inner(qhat, phat)*ds
    + inner(vhat, dot(u('+'), n('+')) - uhat)/eta*dS
    + inner(-vhat, dot(u('-'), n('-')) + uhat)/eta*dS
    + inner(dot(vhat, n), dot(u, n) - uhat)/eta*ds
    )

F = inner(f, q)*dS

params = {
    'ksp_type': 'preonly',
    'mat_type': 'aij',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

w = Function(W)
solve(a==F, w, solver_parameters = params)
