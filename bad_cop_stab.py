from firedrake import *
from firedrake.petsc import PETSc

n = 1
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

def jump(u, n):
    return dot(u('+'), n('+')) + dot(u('-'), n('-'))

n = FacetNormal(mesh)

a = (
    inner(v, kappa*dot(u, J))*dx - inner(div(v), p)*dx 
    + inner(jump(v, n), phat('+'))*dS + inner(dot(v,n), dot(u,n))*ds
    - inner(dot(v('+'), n('+')), dot(u('+'), n('+')) - uhat('+'))/eta*dS
    - inner(dot(v('-'), n('-')), dot(u('-'), n('-')) + uhat('+'))/eta*dS
    - inner(dot(v, n), dot(u, n) - uhat)/eta*ds
    - inner(q, kappa*dot(p, J) +  div(u))*dx
    + inner(qhat('+'), jump(u, n))*dS - inner(qhat, phat)*ds
    + inner(vhat('+'), dot(u('+'), n('+')) - uhat('+'))/eta*dS
    + inner(-vhat('+'), dot(u('-'), n('-')) + uhat('+'))/eta*dS
    + inner(vhat, dot(u, n) - uhat)/eta*ds
    )

F = inner(f, q)*dx

params = {
    'ksp_type': 'preonly',
    'mat_type': 'aij',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

def chop(A, tol=1E-10):
    # remove (near) zeros from sparsity pattern
    A.chop(tol)
    B = PETSc.Mat().create(comm=A.comm)
    B.setType(A.getType())
    B.setSizes(A.getSizes())
    B.setBlockSize(A.getBlockSize())
    B.setUp()
    B.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
    B.setPreallocationCSR(A.getValuesCSR())
    B.assemble()
    A.destroy()
    return B

A = assemble(a)
Amat = chop(A.petscmat)
Amat.view()

w = Function(W)
#solve(a==F, w, solver_parameters = params)
