from firedrake import *
from firedrake.petsc import PETSc


import argparse
parser = argparse.ArgumentParser(description='Stabilised conforming Hdiv badcop')
parser.add_argument('--kappa', type=float, default=1.0, help='kappa')
parser.add_argument('--eta', type=float, default=1.0, help='eta')
parser.add_argument('--damp', action="store_true", default=False, help='damp high frequencies')
parser.add_argument('--nx', type=int, default=10, help='elements along each side')
parser.add_argument('--degree', type=int, default=1, help='degree')
args, _ = parser.parse_known_args()

nx = args.nx
mesh = UnitSquareMesh(nx, nx)

degree = args.degree
RT = FiniteElement("RT", cell=mesh.ufl_cell(), degree=degree)
V = VectorFunctionSpace(mesh, BrokenElement(RT), dim=2)
Q = VectorFunctionSpace(mesh, "DG", degree-1, dim=2)
T = TensorFunctionSpace(mesh, "HDiv Trace", degree-1, shape=(2, 2))
W = V * Q * T

u, p, up_hat = TrialFunctions(W)
v, q, vq_hat = TestFunctions(W)
uhat, phat = up_hat[0], up_hat[1]
vhat, qhat = vq_hat[0], vq_hat[1]

eta = Constant(args.eta)
kappa = Constant(args.kappa)
ikappa = kappa * Constant([[0, -1], [1, 0]])

n = FacetNormal(mesh)
u_n = dot(u, n)
v_n = dot(v, n)

def both(expr):
    return expr('+') + expr('-')

# usual diagonal terms
a = (inner(v, dot(ikappa, u)) * dx
     - inner(q, dot(ikappa, p)) * dx
     + inner(qhat, eta * phat) * ds)

# usual divergence off-diagonal terms
b = lambda q, u: -inner(q, div(u)) * dx
a += b(q, u) + b(p, v)

# constraint off-diagonal terms (with shifted Lagrange multiplier)
c = lambda vhat, qhat, u_n: ( (inner(-vhat('-') + qhat('-'), u_n('-'))
                               +inner(vhat('+') + qhat('+'), u_n('+')) ) * dS
                             + inner(vhat / eta + qhat, u_n) * ds)
a += c(vhat, qhat, u_n) + c(uhat, phat, v_n)

# stabilization diagonal terms
a -= (inner(vhat('+'), uhat('+')) * dS + inner(vhat / eta, uhat) * ds
      + both(inner(v_n, u_n)) * dS + inner(v_n / eta, u_n) * ds)

if args.damp:
    # damping diagonal terms
    du = u_n - uhat
    dv = v_n - vhat
    a -= both(inner(dv, du)) * dS + inner(dv, eta * du) * ds

f = Constant([1,0])
F = inner(f, q)*dx
bcs = []

w = Function(W, name="solution")
for wsub, name in zip(w.subfunctions, ("u", "p")):
    wsub.rename(name)

factor = {
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "petsc",
}

cparams = {
    "ksp_monitor": None,
    "ksp_type": "gmres",
    "pc_type": "python",
    "pc_python_type": "firedrake.ASMStarPC",
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
#solve(a==F, w, bcs=bcs)
problem = LinearVariationalProblem(a, F, w)
solver = LinearVariationalSolver(problem, solver_parameters=sparams)


print = PETSc.Sys.Print
print("dim(W) = ", W.dim(), tuple(V.dim() for V in W))
print("kappa =", float(kappa))
solver.solve()

File("output/bad_cop_stab.pvd").write(*w.subfunctions[:2])

A = assemble(a)


#from firedrake.preconditioners.hypre_ams import chop
#Amat = A.petscmat
#Amat = Amat.convert("seqaij")
#Bmat = chop(Amat)
#Bmat.view()
