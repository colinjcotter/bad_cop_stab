from firedrake import *
from firedrake.petsc import PETSc

nx = 10
mesh = UnitSquareMesh(nx, nx, reorder=False, distribution_parameters={"partition": False})

degree = 1
element = BrokenElement(FiniteElement("RT", cell=mesh.ufl_cell(), degree=degree))
V = VectorFunctionSpace(mesh, element, dim=2)
Q = VectorFunctionSpace(mesh, "DG", degree-1, dim=2)
T = VectorFunctionSpace(mesh, "HDiv Trace", degree-1, dim=2)
W = V * Q * T * T

u, p, uhat, phat = TrialFunctions(W)
v, q, vhat, qhat = TestFunctions(W)


eta = Constant(1.0)
kappa = Constant(0.0)
ikappa = kappa * Constant([[0, -1], [1, 0]])

n = FacetNormal(mesh)
u_n = dot(u, n)
v_n = dot(v, n)


# diagonal terms
a1 = inner(v, dot(ikappa, u)) * dx - (inner(v_n('+') * (1/eta), u_n('+')) + inner(v_n('-') * (1/eta), u_n('-'))) * dS - inner(v_n * (1/eta), u_n) * ds
a2 = -inner(q, dot(ikappa, p)) * dx
a3 = -inner(vhat('+') * (1/eta), uhat('+')) * dS - inner(vhat * (1/eta), uhat) * ds
a4 = inner(qhat, eta * phat) * ds
a4 = 0


# off-diagonal terms
b = lambda q, u: -inner(q, div(u)) * dx
c = lambda vhat, u_n: (inner(vhat('+') * (1/eta), u_n('+') - u_n('-'))) * dS + inner(vhat * (1/eta), u_n) * ds
d = lambda qhat, u_n: (inner(qhat('+'), u_n('+') + u_n('-'))) * dS + inner(qhat, u_n) * ds

a = a1 + a2 + a3 + a4 + b(q, u) + b(p, v) + c(vhat, u_n) + c(uhat, v_n) + d(qhat, u_n) + d(phat, v_n)

if False:
    # damping term
    du = u - uhat
    dv = v - vhat
    a += (inner(dv('+'), eta * du('+')) + inner(dv('-'), eta * du('-'))) * dS + inner(dv, eta * du) * ds


f = Constant([1,0])
F = inner(f, q)*dx
bcs = []

w = Function(W, name="solution")
for wsub, name in zip(w.subfunctions, ("u", "p")):
    wsub.rename(name)

solve(a==F, w, bcs=bcs)

File("output/bad_cop_stab.pvd").write(*w.subfunctions[:2])

A = assemble(a)

PETSc.Sys.Print(tuple(V.dim() for V in W))

#from firedrake.preconditioners.hypre_ams import chop
#Amat = A.petscmat
#Amat = Amat.convert("seqaij")
#Bmat = chop(Amat)
#Bmat.view()
