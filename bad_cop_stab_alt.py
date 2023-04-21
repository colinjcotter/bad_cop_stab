from firedrake import *
from firedrake.petsc import PETSc

nx = 10
mesh = UnitSquareMesh(nx, nx, reorder=False, distribution_parameters={"partition": False})

degree = 1
RT = FiniteElement("RT", cell=mesh.ufl_cell(), degree=degree)
V = VectorFunctionSpace(mesh, BrokenElement(RT), dim=2)
Q = VectorFunctionSpace(mesh, "DG", degree-1, dim=2)
T = TensorFunctionSpace(mesh, "HDiv Trace", degree-1, shape=(2, 2))
W = V * Q * T

u, p, up_hat = TrialFunctions(W)
v, q, vq_hat = TestFunctions(W)
uhat, phat = up_hat[0], up_hat[1]
vhat, qhat = vq_hat[0], vq_hat[1]

eta = Constant(1.0)
kappa = Constant(1.0)
ikappa = kappa * Constant([[0, -1], [1, 0]])

n = FacetNormal(mesh)
u_n = dot(u, n)
v_n = dot(v, n)

both = lambda f: f('+') + f('-')

# diagonal terms
a1 = inner(v, dot(ikappa, u)) * dx - both(inner(v_n / eta, u_n)) * dS - inner(v_n * (1/eta), u_n) * ds
a2 = -inner(q, dot(ikappa, p)) * dx
a3 = -inner(vhat('+') * (1/eta), uhat('+')) * dS - inner(vhat * (1/eta), uhat) * ds
a4 = inner(qhat, eta * phat) * ds

# off-diagonal terms
b = lambda q, u: -inner(q, div(u)) * dx
c = lambda vhat, qhat, u_n: ( (inner(-vhat('-') / eta + qhat('-'), u_n('-'))
                              + inner(vhat('+') / eta + qhat('+'), u_n('+'))) * dS
                             + inner(vhat / eta + qhat, u_n) * ds)
a = a1 + a2 + a3 + a4 + b(q, u) + b(p, v) + c(vhat, qhat, u_n) + c(uhat, phat, v_n)

if False:
    # damping term
    du = u - uhat
    dv = v - vhat
    a += both(inner(dv, eta * du)) * dS + inner(dv, eta * du) * ds


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
