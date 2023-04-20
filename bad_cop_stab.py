from firedrake import *
from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

import argparse
parser = argparse.ArgumentParser(description='Stabilised conforming Hdiv badcop')
parser.add_argument('--kappa', type=float, default=1.0, help='kappa')
args = parser.parse_known_args()
args = args[0]


nx = 10
mesh = UnitSquareMesh(nx, nx, reorder=False, distribution_parameters={"partition": False})

degree = 1
Vele = BrokenElement(FiniteElement("RT", mesh.ufl_cell(), degree, variant="integral"))
V = VectorFunctionSpace(mesh, Vele, dim=2)
Q = VectorFunctionSpace(mesh, "DG", degree-1, dim=2)
eta = Constant(1.0)
kappa = Constant(args.kappa)

f = as_vector([Constant(1.0), Constant(0.0)])

trace = False
if trace:
    T = VectorFunctionSpace(mesh, "HDiv Trace", degree-1, dim=4)
    W = V * Q * T 

    u, p, tt = TrialFunctions(W)
    v, q, ss = TestFunctions(W)
    
    uhat = as_vector([tt[0], tt[1]])
    phat = as_vector([tt[2], tt[3]])
    vhat = as_vector([ss[0], ss[1]])
    qhat = as_vector([ss[2], ss[3]])
else:
    Vhat = VectorFunctionSpace(mesh, "RT", degree, dim=2)
    W = V * Q * Vhat * Vhat
    u, p, uhat, phat = TrialFunctions(W)
    v, q, vhat, qhat = TestFunctions(W)
    
J = as_tensor([[0., -1.], [1., 0.]])

def jump(u, n):
    return dot(u('+'), n('+')) + dot(u('-'), n('-'))

def both(q):
    return q('+') + q('-')

n = FacetNormal(mesh)

if trace:
    a = (
        - inner(q, kappa*dot(J, p) +  div(u))*dx
        + inner(v, kappa*dot(J, u))*dx - inner(div(v), p)*dx
        + inner(jump(v, n), phat('+'))*dS
        + inner(dot(v,n), phat)*ds
        + inner(qhat('+'), jump(u, n))*dS
        + inner(qhat,dot(u, n))*ds
        - inner(dot(v('+'), n('+')) - vhat('+'), dot(u('+'), n('+')) - uhat('+'))/eta*dS
        - inner(dot(v('-'), n('-')) + vhat('+'), dot(u('-'), n('-')) + uhat('+'))/eta*dS
        - inner(dot(v, n) - vhat, dot(u, n) - uhat)/eta*ds
        + inner(qhat, phat)*ds
    )
else:
    phatt = dot(phat, n)
    qhatt = dot(qhat, n)
    a = (
        - inner(q, kappa*dot(J, p) +  div(u))*dx
        + inner(v, kappa*dot(J, u))*dx - inner(div(v), p)*dx
        + inner(jump(v, n), phatt('+'))*dS
        + inner(dot(v,n), phatt)*ds
        + inner(qhatt('+'), jump(u, n))*dS
        + inner(qhatt, dot(u, n))*ds
        - both(inner(dot(v, n) - dot(vhat,n), dot(u, n) - dot(uhat, n)))/eta*dS
        - inner(dot(v, n) - dot(vhat, n), dot(u, n) - dot(uhat, n))/eta*ds
        + inner(qhatt, phatt)*ds
    )

F = inner(f, q)*dx

params = {
    'ksp_type': 'preonly',
    'mat_type': 'aij',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

hparams = {
    'ksp_type': 'preonly',
    'ksp_monitor': None,
    'mat_type': 'matfree',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.SCPC',
    'pc_sc_eliminate_fields' : '0,1',
    'condensed_field_ksp_type' : 'preonly',
    'condensed_field_pc_type' : 'lu'
}

starparams = {
    'ksp_type': 'gmres',
    'ksp_monitor': None,
    'pc_type': 'python',
    "pc_python_type": "firedrake.ASMStarPC",
    "pc_star_construct_dim": 0,
    "pc_star_sub_sub_pc_type": "lu",
    "pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack",
}


factor = {
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "petsc",
}

cparams = {
    "ksp_monitor": None,
    "ksp_type": "gmres",
    "pc_type": "python",
    "pc_python_type": "firedrake.ASMStarPC",
    #"pc_star_mat_ordering_type": "metisnd",
    "pc_star_construct_dim": 0,
    "pc_star_sub_sub": factor, # the first sub is PCASM and second is subsolver
}

sparams = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "mat_type": "matfree",
    "pc_python_type": "firedrake.SCPC",
    "pc_sc_eliminate_fields": "0,1",
    "condensed_field": cparams,
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

#from numpy import *
#A = array(assemble(a).M.values)
#import matplotlib.pyplot as pp
#pp.pcolor(A)
#pp.colorbar()
#pp.show()
w = Function(W)
solve(a==F, w, solver_parameters = starparams)

if trace:
    u, p, tt = w.split()
else:
    u, p, uhat, phat = w.split()

File('helm.pvd').write(u, p)
