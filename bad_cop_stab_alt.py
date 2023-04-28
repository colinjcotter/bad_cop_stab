from firedrake import *
from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()
print = PETSc.Sys.Print

import argparse
parser = argparse.ArgumentParser(description='Stabilised conforming Hdiv badcop')
parser.add_argument('--kappa', type=float, default=1.0, help='kappa')
parser.add_argument('--eta', type=float, default=1.0, help='eta')
parser.add_argument('--cr', action="store_true", default=False, help='use Crouzeix--Raviart instead of HDiv Trace')
parser.add_argument('--no-stab', dest="stab", action="store_false", default=True, help='do not use stablization')
parser.add_argument('--no-mg', dest="mg", action="store_false", default=True, help='do not use multigrid')
parser.add_argument('--no-slate', dest="slate", action="store_false", default=True, help='do not use slate')
parser.add_argument('--no-pressure', dest="pressure", action="store_false", default=True, help='eliminate pressure')
parser.add_argument('--complex', action="store_true", default=False, help='use complex mode')
parser.add_argument('--quadrilateral', action="store_true", default=False, help='use tensor-product cells')
parser.add_argument('--nx', type=int, default=10, help='elements along each side')
parser.add_argument('--refine', type=int, default=0, help='level of refinement')
parser.add_argument('--degree', type=int, default=1, help='degree')
args, _ = parser.parse_known_args()

stab = args.stab
pressure = args.pressure
complex_mode = args.complex
use_mg = args.mg
use_slate = args.slate
dist_params = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1),}

nx = args.nx
mesh = UnitSquareMesh(nx, nx, quadrilateral=args.quadrilateral, distribution_parameters=dist_params)
mh = MeshHierarchy(mesh, args.refine)
mesh = mh[-1]

eta = Constant(args.eta, domain=mesh)
kappa = Constant(args.kappa, domain=mesh)

degree = args.degree
cell = mesh.ufl_cell()
is_simplex = cell.is_simplex()
if is_simplex:
    rt_family = "RT"
    dg_family = "DG"
else:
    rt_family = "RTCF" if cell.topological_dimension() == 2 else "NCF"
    dg_family = "DQ"

RT = FiniteElement(rt_family, cell=cell, degree=degree)
DG = FiniteElement(dg_family, cell=cell, degree=degree-1)
if args.cr:
    T = FiniteElement("CR", cell=cell, degree=RT.degree())
    if T.degree() != 1:
        T = T["facet"]
else:
    T = FiniteElement("HDiv Trace", cell=cell, degree=DG.degree())

BrokenRT = BrokenElement(RT)
if complex_mode:
    j = 1j
    if stab:
        T = VectorElement(T, dim=2)
else:
    j = as_tensor([[0, -1], [1, 0]])
    BrokenRT = VectorElement(BrokenRT, dim=2)
    DG = VectorElement(DG, dim=2)
    if stab:
        T = TensorElement(T, shape=(2, 2))
    else:
        T = VectorElement(T, dim=2)

if pressure:
    elements = (BrokenRT, DG, T)
else:
    elements = (BrokenRT, T)
element = MixedElement(elements)
W = FunctionSpace(mesh, element)
w = Function(W, name="solution")

names = ("u", "p", "traces")
if len(W) == 2:
    names = names[::2]
for wsub, name in zip(w.subfunctions, names):
    wsub.rename(name)

if len(W) == 3:
    u, p, up_hat = TrialFunctions(W)
    v, q, vq_hat = TestFunctions(W)
else:
    u, up_hat = TrialFunctions(W)
    v, vq_hat = TestFunctions(W)
    # these are off by a factor of i, but compensated below by flipping a sign
    p = div(u) / kappa
    q = div(v) / kappa

if stab:
    if up_hat.ufl_shape[0] == 2:
        uhat, phat = up_hat[0], up_hat[1]
        vhat, qhat = vq_hat[0], vq_hat[1]
    elif up_hat.ufl_shape[0] == 4:
        _up_hat = tuple(up_hat[i] for i in range(4))
        _vq_hat = tuple(vq_hat[i] for i in range(4))
        uhat, phat = as_vector(_up_hat[0:2]), as_vector(_up_hat[2:4])
        vhat, qhat = as_vector(_vq_hat[0:2]), as_vector(_vq_hat[2:4])
else:
    phat = up_hat
    qhat = vq_hat

if not complex_mode:
    # Complex conjugate the test functions
    C = diag(as_vector([1, -1]))
    if stab:
        vhat = C * vhat
    qhat = C * qhat
    v = C * v
    q = C * q

def both(expr):
    return expr('+') + expr('-')

n = FacetNormal(mesh)
u_n = dot(u, n)
v_n = dot(v, n)

dx0 = dx(degree=2*degree+1, domain=mesh)
dx1 = dx(degree=2*degree-1, domain=mesh)
ds1 = ds(degree=2*degree-1, domain=mesh)
dS1 = dS(degree=2*degree-1, domain=mesh)

ikappa = kappa * j

# usual diagonal terms
a = (- inner(v, ikappa * u) * dx0
     + inner(q, ikappa * p) * dx1
     - inner(qhat, eta * phat) * ds1)

if len(W) == 3:
    # usual divergence off-diagonal terms
    a += (inner(q, div(u)) + inner(div(v), p)) * dx1

# usual constraint off-diagonal terms
a -= both(inner(qhat, u_n)) * dS1 + inner(qhat, u_n) * ds1
a -= both(inner(phat, v_n)) * dS1 + inner(phat, v_n) * ds1

# stabilization terms
if stab:
    r = [Function(FunctionSpace(m, "RT", 1)) for m in mh]
    for ri in r:
        ri.assign(1)
    for k in range(1, len(r)):
        r[k]._coarse = r[k-1]

    s = dot(r[-1], n)
    v_minus = v_n - s * vhat
    u_minus = u_n - s * uhat
    a += (both(inner(v_minus, u_minus)) * dS1
          + inner(v_minus, (1/eta) * u_minus) * ds1)

# Right-hand side
f = Constant(1 if complex_mode else [1, 0], domain=mesh)
g = zero(qhat.ufl_shape)

x = SpatialCoordinate(mesh)
omega = [Constant(2*pi, domain=mesh) for _ in x]
p_exact = Constant(1 if complex_mode else [1,0], domain=mesh) * (x[0]*(1-x[0]) * x[1]*(1-x[1])/0.5**4)**2

u_exact = None
# p_exact = None
if p_exact:
    u_exact = (ikappa / kappa**2) * grad(p_exact)
    f = ikappa * p_exact + div(u_exact)
    g = eta * p_exact + dot(u_exact, n)

if len(W) == 2:
    f = j * f
F = inner(q, f) * dx1 - inner(qhat, g) * ds1
bcs = []

gmg = lambda coarse, levels: {
    "pc_type": "mg",
    "mg_coarse": coarse,
    "mg_levels": levels,
}

factor = lambda solver="petsc": {
    "pc_type": "cholesky",
    "pc_factor_mat_solver_type": solver,
}

levels = {
    "ksp_type": "chebyshev",
    "ksp_chebyshev_kind": "fourth",
    "pc_type": "python",
    "pc_python_type": "firedrake.ASMStarPC",
    "pc_star_construct_dim": 0,
    "pc_star_backend": "tinyasm",
}

sparams = factor("mumps")
if use_mg:
    sparams = gmg(sparams, levels) if args.refine else levels

sparams.update({
    "mat_type": "aij",
    "ksp_monitor": None,
    "ksp_view_eigenvalues": None,
    "ksp_type": "gmres",
    "ksp_pc_side": "right",
    "ksp_norm_type": "unpreconditioned",
})

if use_slate:
    sparams = {
        "ksp_type": "preonly",
        "snes_monitor": None,
        "pc_type": "python",
        "mat_type": "matfree",
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": ",".join(map(str, range(len(W)-1))),
        "condensed_field": sparams,
    }

problem = LinearVariationalProblem(a, F, w, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=sparams, options_prefix="")

print("dim(W) =", W.dim(), tuple(V.dim() for V in W))
print("kappa =", float(kappa))
solver.solve()

uh = w.subfunctions[0]
if len(W) == 3:
    ph = w.subfunctions[1]
else:
    Q = FunctionSpace(mesh, DG)
    ph = Function(Q, name="p")
    ph.interpolate((ikappa*div(uh) - kappa*f)/kappa**2)

if p_exact:
    u_diff = uh - u_exact
    p_diff = ph - p_exact
    error = sqrt(assemble(inner(u_diff, u_diff) * dx0 + inner(p_diff, p_diff) * dx1))
    print("error", error)

File("output/bad_cop_stab.pvd").write(uh, ph)
