from firedrake import *
from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()
print = PETSc.Sys.Print

import argparse
parser = argparse.ArgumentParser(description='Stabilised conforming Hdiv badcop')
parser.add_argument('--kappa', type=float, default=1.0, help='kappa')
parser.add_argument('--eta', type=float, default=1.0, help='eta')
parser.add_argument('--cr', action="store_true", default=False, help='use Crouzeix--Raviart instead of HDiv Trace')
parser.add_argument('--complex', action="store_true", default=False, help='use complex mode')
parser.add_argument('--quadrilateral', action="store_true", default=False, help='use tensor-product cells')
parser.add_argument('--nx', type=int, default=10, help='elements along each side')
parser.add_argument('--refine', type=int, default=0, help='level of refinement')
parser.add_argument('--degree', type=int, default=1, help='degree')
args, _ = parser.parse_known_args()

complex_mode = args.complex
dist_params = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1),}

nx = args.nx
mesh = UnitSquareMesh(nx, nx, quadrilateral=args.quadrilateral, distribution_parameters=dist_params)
mh = MeshHierarchy(mesh, args.refine)
mesh = mh[-1]

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
uhat, phat = up_hat[0], up_hat[1]
vhat, qhat = vq_hat[0], vq_hat[1]

n = FacetNormal(mesh)
u_n = dot(u, n)
v_n = dot(v, n)

eta = Constant(args.eta)
kappa = Constant(args.kappa)
ikappa = kappa * j

def minus(expr):
    return expr('-')

def plus(expr):
    return expr('+')

def both(expr):
    return expr('+') + expr('-')

dx0 = dx(degree=2*degree+1)
dx1 = dx(degree=2*degree-1)
ds = ds(degree=2*degree-1)
dS = dS(degree=2*degree-1)

# usual diagonal terms
a = (- inner(v, ikappa * u) * dx0
     + inner(q, ikappa * p) * dx1
     - inner(qhat, eta * phat) * ds)

# usual divergence off-diagonal terms
a += (inner(q, div(u)) + inner(div(v), p)) * dx1

# usual constraint off-diagonal terms
c = lambda qhat, u_n: both(inner(qhat, u_n)) * dS + inner(qhat, u_n) * ds
a -= c(qhat, u_n) + c(phat, v_n)

# stabilization terms
a += ((minus(inner(v_n + vhat, u_n + uhat))
       + plus(inner(v_n - vhat, u_n - uhat))) * dS
      + inner(v_n - vhat, (1/eta) * (u_n - uhat)) * ds)

# Right-hand side
f = Constant(1 if complex_mode else [1, 0])
F = inner(q, f) * dx
bcs = []

w = Function(W, name="solution")
for wsub, name in zip(w.subfunctions, ("u", "p", "traces")):
    wsub.rename(name)

gmg = lambda coarse, levels: {
    "pc_type": "mg",
    "mg_levels": levels,
    "mg_coarse": coarse,
}

factor = lambda solver="petsc": {
    "pc_type": "lu",
    "pc_factor_mat_solver_type": solver,
}

levels = {
    "ksp_type": "chebyshev",
    "pc_type": "python",
    "pc_python_type": "firedrake.ASMStarPC",
    "pc_star_construct_dim": 0,
    "pc_star_sub_sub": factor("petsc"), # the first sub is PCASM and second is subsolver
}

cparams = gmg(factor("mumps"), levels) if args.refine else levels
cparams.update({
    "mat_type": "aij",
    "ksp_monitor": None,
    "ksp_type": "gmres",
})

sparams = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "mat_type": "matfree",
    "pc_python_type": "firedrake.SCPC",
    "pc_sc_eliminate_fields": "0,1",
    "condensed_field": cparams,
}
problem = LinearVariationalProblem(a, F, w)
solver = LinearVariationalSolver(problem, solver_parameters=sparams, options_prefix="")

print("dim(W) = ", W.dim(), tuple(V.dim() for V in W))
print("kappa =", float(kappa))
solver.solve()
File("output/bad_cop_stab.pvd").write(*w.subfunctions[:2])
