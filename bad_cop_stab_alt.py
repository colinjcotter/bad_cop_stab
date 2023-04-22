from firedrake import *
from firedrake.petsc import PETSc


import argparse
parser = argparse.ArgumentParser(description='Stabilised conforming Hdiv badcop')
parser.add_argument('--kappa', type=float, default=1.0, help='kappa')
parser.add_argument('--eta', type=float, default=1.0, help='eta')
parser.add_argument('--cr', action="store_true", default=False, help='use Crouzeix--Raviart instead of HDiv Trace')
parser.add_argument('--damp', action="store_true", default=False, help='damp high frequencies')
parser.add_argument('--complex', action="store_true", default=False, help='use complex mode')
parser.add_argument('--quadrilateral', action="store_true", default=False, help='use tensor-product cells')
parser.add_argument('--nx', type=int, default=10, help='elements along each side')
parser.add_argument('--degree', type=int, default=1, help='degree')
args, _ = parser.parse_known_args()

complex_mode = args.complex
dist_params = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1),}

nx = args.nx
mesh = UnitSquareMesh(nx, nx, quadrilateral=args.quadrilateral, distribution_parameters=dist_params)

degree = args.degree
cell = mesh.ufl_cell()
is_simplex = cell.is_simplex()

RT = FiniteElement("RT" if is_simplex else "RTCF", cell=cell, degree=degree)
DG = FiniteElement("DG" if is_simplex else "DQ", cell=cell, degree=degree-1)
if args.cr:
    T = FiniteElement("CR", cell=cell, degree=degree)
else:
    T = FiniteElement("HDiv Trace", cell=cell, degree=degree-1)

BrokenRT = BrokenElement(RT)
if complex_mode:
    j = Constant(1j)
    T = VectorElement(T, dim=2)
else:
    j = Constant([[0, -1], [1, 0]])
    BrokenRT = VectorElement(BrokenRT, dim=2)
    DG = VectorElement(DG, dim=2)
    T = TensorElement(T, shape=(2, 2))

W = FunctionSpace(mesh, MixedElement([BrokenRT, DG, T]))
u, p, up_hat = TrialFunctions(W)
v, q, vq_hat = TestFunctions(W)

n = FacetNormal(mesh)
u_n = dot(u, n)
v_n = dot(v, n)

uhat, phat = up_hat[0], up_hat[1]
vhat, qhat = vq_hat[0], vq_hat[1]

eta = Constant(args.eta)
kappa = Constant(args.kappa)
ikappa = kappa * j
inv_eta = 1 / eta

def minus(expr):
    return expr('-')

def plus(expr):
    return expr('+')

def both(expr):
    return expr('+') + expr('-')

# usual diagonal terms
a = (- inner(v, ikappa * u) * dx
     + inner(q, ikappa * p) * dx
     - inner(qhat, eta * phat) * ds)

# usual divergence off-diagonal terms
a += (inner(q, div(u)) + inner(div(v), p)) * dx

# constraint off-diagonal terms (with shifted Lagrange multiplier)
c = lambda vhat, qhat, u_n: ((minus(inner(qhat - vhat, u_n))
                              + plus(inner(qhat + vhat, u_n))) * dS
                             + inner(qhat + inv_eta * vhat, u_n) * ds)

a -= c(vhat, qhat, u_n) + c(uhat, phat, v_n)

# stabilization diagonal terms
a += both(inner(v_n, u_n)) * dS + inner(inv_eta * v_n, u_n) * ds
a += plus(inner(vhat, uhat)) * dS + inner(inv_eta * vhat, uhat) * ds

if args.damp:
    # damping diagonal terms
    du = u_n - uhat
    dv = v_n - vhat
    a += both(inner(dv, du)) * dS + inner(dv, eta * du) * ds

f = Constant(1 if complex_mode else [1, 0])
F = inner(q, f) * dx
bcs = []

w = Function(W, name="solution")
for wsub, name in zip(w.subfunctions, ("u", "p", "traces")):
    wsub.rename(name)

factor = {
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "petsc",
}

cparams = {
    "ksp_monitor": None,
    "ksp_type": "gmres",
    "ksp_view_eigenvalues": None,
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
problem = LinearVariationalProblem(a, F, w)
solver = LinearVariationalSolver(problem, solver_parameters=sparams)

print = PETSc.Sys.Print
print("dim(W) = ", W.dim(), tuple(V.dim() for V in W))
print("kappa =", float(kappa))
solver.solve()

File("output/bad_cop_stab.pvd").write(*w.subfunctions[:2])

#from firedrake.preconditioners.hypre_ams import chop
#A = assemble(a)
#Amat = A.petscmat
#Amat = Amat.convert("seqaij")
#Bmat = chop(Amat)
#Bmat.view()
