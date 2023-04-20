from firedrake.petsc import PETSc
from firedrake.mg.utils import get_level
from firedrake.preconditioners.fdm import tabulate_exterior_derivative
from functools import partial
import firedrake.dmhooks as dmhooks
import ufl
import firedrake
import numpy


mg_log = False
mg_log = True


def as_mixed_function(expressions):
    components = []
    for x in expressions:
        components.extend(x[idx] for idx in numpy.ndindex(x.ufl_shape))
    return ufl.as_vector(components)


def interior_facet_measure(mesh, degree=None):
    meta = {"domain": mesh, "degree": degree,}
    if mesh.cell_set._extruded:
        return firedrake.dS_v(**meta) + firedrake.dS_h(**meta)
    else:
        return firedrake.dS(**meta)


def exterior_facet_measure(mesh, subs, degree=None):
    meta = {"domain": mesh, "degree": degree,}
    if mesh.cell_set._extruded:
        ds_ext = {"on_boundary": firedrake.ds_v(**meta),
                  "bottom": firedrake.ds_b(**meta),
                  "top": firedrake.ds_t(**meta)}
        return [ds_ext.get(s) or firedrake.ds_v(s, **meta) for s in subs]
    else:
        ds_ext = {"on_boundary": firedrake.ds(**meta)}
        return [ds_ext.get(s) or firedrake.ds(s, **meta) for s in subs]


def unique_facet_markers(mesh):
    return list(set(mesh.exterior_facets.unique_markers) | set(mesh.interior_facets.unique_markers))


def complement_boundary_markers(mesh, subs=[]):
    if not subs:
        csubs = ["on_boundary"]
    elif "on_boundary" in subs:
        csubs = []
    else:
        make_tuple = lambda s: s if type(s) == tuple else (s,)
        csubs = list(set(mesh.exterior_facets.unique_markers) - set(sum([make_tuple(s) for s in subs if type(s) != str], ())))

    if mesh.cell_set._extruded:
        if not "top" in subs:
            csubs.append("top")
        if not "bottom" in subs:
            csubs.append("bottom")
    return csubs


def sipg_energy(u, u_bc, F_v, eta, dirichlet_ids, quad_degree=None):
    mesh = u.ufl_domain()
    n = ufl.FacetNormal(mesh)
    if mesh.topological_dimension() == 1:
        hinv = abs(1/ufl.JacobianDeterminant(mesh))
    else:
        hinv = abs(ufl.geometry.CellFacetJacobianDeterminant(mesh)/ufl.JacobianDeterminant(mesh))

    #hinv = ufl.FacetArea(mesh)/ufl.CellVolume(mesh)
    penalty = eta*hinv

    dS_int = interior_facet_measure(mesh, degree=quad_degree)
    ds_Dir = exterior_facet_measure(mesh, dirichlet_ids, degree=quad_degree)

    flux_u = F_v(u, ufl.grad(u))
    jump_u = ufl.outer(u("-"), n("-")) + ufl.outer(u("+"), n("+"))
    num_flux = F_v(ufl.avg(u), ufl.avg(penalty/2)*jump_u)
    U = ufl.inner(num_flux-ufl.avg(flux_u), jump_u)*dS_int

    try:
        u_bc.ufl_shape
        u_bc = [u_bc]
        ds_Dir = [sum(ds_Dir, ufl.ds(tuple()))]
    except AttributteError:
        pass

    for u0, ds0 in zip(u_bc, ds_Dir):
        jump_u = ufl.outer(u-u0, n)
        num_flux = F_v(u, (penalty/2)*jump_u)
        U += ufl.inner(num_flux-flux_u, jump_u)*ds0
    return U


class MassPC(firedrake.AssembledPC):
    """
    Mass preconditioner that can be updated
    """
    _prefix = "Mp_"
    def form(self, pc, test, trial):
        _, P = pc.getOperators()
        context = P.getPythonContext()
        mu = context.appctx.get("mu", 1.0)

        V = test.function_space()
        degree = firedrake.PMGPC.max_degree(V.ufl_element())
        quad_degree = 3*(degree+1)-1
        dx = ufl.dx(degree=quad_degree, domain=V.mesh())
        aP = ufl.inner((1/mu)*test, trial)*dx
        bcs = []
        return aP, bcs

massinv = lambda pc :{
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": __name__+".MassPC",
    "Mp": pc,
}

ksp_pc = lambda ksp: {
    "pc_type": "ksp",
    "ksp": ksp,
}

pcnone = lambda : {
    "ksp_type": "preonly",
    "pc_type": "none",
}


jacobi = lambda : {
    "ksp_type": "preonly",
    "pc_type": "jacobi",
}

pbjacobi = lambda : {
    "ksp_type": "preonly",
    "pc_type": "pbjacobi",
}

bjacobi = lambda sub, use_amat=False: {
    "ksp_type": "preonly",
    "pc_type": "bjacobi",
    "pc_use_amat": use_amat,
    "sub": sub,
}

rich = lambda its, scale=0.25: {
    "ksp_max_it": its,
    "ksp_type": "richardson",
    "ksp_norm_type": "NONE",
    "ksp_convergence_test": "skip",
    "ksp_richardson_scale": scale,
}

cheb = lambda its, scale=0.125, **kwargs: {
    "ksp_max_it": its,
    "ksp_type": "chebyshev",
    "ksp_norm_type": "NONE",
    "ksp_convergence_test": "skip",
    "esteig_ksp_type": kwargs.get("esteig_ksp_type", "cg"),
    "esteig_ksp_pc_side": kwargs.get("esteig_ksp_pc_side", "left"),
    "esteig_ksp_norm_type": kwargs.get("esteig_ksp_norm_type", "natural"),
    "esteig_ksp_view_eigenvalues": None,
    "esteig_ksp_atol": 0E-8,
    "esteig_ksp_rtol": 1E-8,
    "esteig_ksp_max_it": 14,
    "ksp_chebyshev_esteig_steps": 14,
    "ksp_chebyshev_esteig_noisy": True,
    "ksp_chebyshev_esteig": kwargs.get("esteig", "%f,%f,%f,%f" % (1-3*scale/2, scale/2, scale/2, 1+scale/2)),
}

ssor = lambda omega: {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "eisenstat",
    "pc_eisenstat_omega": omega,
    "pc_factor_mat_ordering_type": "natural",
}

icc = lambda levels, in_place=True: {
    "mat_type": "sbaij",
    "ksp_type": "preonly",
    "pc_type": "icc",
    "pc_factor_levels": levels,
    "pc_factor_in_place": in_place,
    "pc_factor_mat_ordering_type": "natural",
}

ilu = lambda levels, in_place=True: {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "ilu",
    "pc_factor_levels": levels,
    "pc_factor_in_place": in_place,
    "pc_factor_mat_ordering_type": "natural",
}

chol = lambda solver="petsc", ordering_type="natural": {
    "mat_type": "sbaij",
    "ksp_type": "preonly",
    "pc_type": "cholesky",
    "pc_factor_mat_solver_type": solver,
    "pc_factor_mat_ordering_type": ordering_type,
    "pc_factor_in_place": solver == "petsc",
    "pc_factor_reuse_fill": solver == "petsc",
    #"pc_factor_shift_type": "nonzero",
    #"pc_factor_shift_amount": 1E-10 if solver == "mumps" else 0.0E0,
    #"mat_cholmod_dbound": 1E-10,
}

lu = lambda solver="petsc", ordering_type="natural": {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": solver,
    "pc_factor_mat_ordering_type": ordering_type,
    "pc_factor_in_place": solver == "petsc",
    "pc_factor_reuse_fill": solver == "petsc",
    #"pc_factor_shift_type": "nonzero",
    #"pc_factor_shift_amount": 1E-10 if solver == "mumps" else 0.0E0,
}

telescope = lambda pc: {
    "mat_type": pc.get("mat_type", "aij"),
    "ksp_type": "preonly",
    "pc_type": "telescope",
    "pc_telescope_reduction_factor": 1,
    "telescope": pc,
}

star = lambda dim, pc: {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.ASMExtrudedStarPC",
    "pc_star_mat_ordering_type": "metisnd",
    "pc_star_construct_dim": dim,
    "pc_star_sub_sub": pc, # the first sub is PCASM and second is subsolver
}

facetsplit = lambda pc: {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.FacetSplitPC",
    "facet": pc,
}

assembled = lambda pc, **kwargs: {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled": dict(pc, **kwargs),
}

fdm = lambda pc, **kwargs: {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.FDMPC",
    "fdm": dict(pc, **kwargs),
}

tinyasm = lambda dim: {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.ASMExtrudedStarPC",
    "pc_star_backend": "tinyasm",
    "pc_star_construct_dim": dim,
}

pcpatch = lambda dim: {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch": {
        "pc_patch": {
            "save_operators": True,
            "partition_of_unity": False,
            "construct_type": "star",
            "construct_dim": dim,
            "sub_mat_type": "seqdense",
            "dense_inverse": True,
            "precompute_element_tensors": None},
        "sub": {
            "ksp_type": "preonly",
            "pc_type": "lu",}},
}

hypre = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

hypre_ams = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.HypreAMS",
}

hypre_ads = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.HypreADS",
}

gmg = lambda coarse, levels, mg_type="multiplicative", cycles=1, cycle_type="v": {
    "ksp_type": "preonly",
    "pc_type": "mg",
    "mg_levels": levels,
    "mg_coarse": coarse,
    "pc_mg_type": mg_type,
    "pc_mg_cycle_type": cycle_type,
    "pc_mg_multiplicative_cycles": cycles,
    "pc_mg_log": mg_log,
}

pmg = lambda coarse, levels, mg_type="multiplicative", cycles=1, cycle_type="v": {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.P1PC",
    "pmg_mg_coarse_degree": 1,
    "pmg_pc_mg_type": mg_type,
    "pmg_pc_mg_cycle_type": cycle_type,
    "pmg_pc_mg_multiplicative_cycles": cycles,
    "pmg_mg_coarse": coarse or chol("mumps"),
    "pmg_mg_levels": levels,
    "pmg_pc_mg_log": mg_log,
}

hiptmair = lambda coarse, levels, mg_type="additive": {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.HiptmairPC",
    "hiptmair_pc_mg_type": mg_type,
    "hiptmair_mg_levels": levels,
    "hiptmair_mg_coarse": coarse,
    "hiptmair_pc_mg_log": mg_log,
}

schur = lambda pc_0, pc_1, schur_type="full", scale=-1, ptype="a11": {
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": schur_type,
    "pc_fieldsplit_schur_scale": scale,
    "fieldsplit_schur_precondition": ptype,
    "fieldsplit_0": pc_0,
    "fieldsplit_1": pc_1,
}

def bdiag(*args, fieldsplit_type=None):
    fs = {
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": fieldsplit_type or "additive",
    }
    for j, pc in enumerate(args):
        fs["fieldsplit_%d" % j] = pc
    return fs


def star_relax(edim, factor=None):
    # It does not make sense to solve stars with Jacobi
    if factor and factor["pc_type"].endswith("jacobi"):
        return factor
    else:
        return star(edim, factor or chol("cholmod"))


def gmg_afw(formdegree, mg_type, coarse, steps=1, cycles=1, cycle_type="v", **kwargs):
    from firedrake.preconditioners.asm import have_tinyasm
    relax = tinyasm if have_tinyasm else pcpatch

    #relax = star_relax
    #relax = pcpatch
    levels = relax(0) if formdegree else jacobi()
    if mg_type != "additive":
        levels.update(cheb(steps, **kwargs))
    return gmg(coarse, levels, mg_type=mg_type, cycles=cycles, cycle_type=cycle_type)


def gmg_hiptmair(formdegree, mg_type, coarse, steps=1, cycles=1, cycle_type="v", **kwargs):
    levels = jacobi()
    if formdegree:
        levels = hiptmair(levels, levels, mg_type="additive")
    if mg_type != "additive":
        levels.update(cheb(steps, **kwargs))
    return gmg(coarse, levels, mg_type=mg_type, cycles=cycles, cycle_type=cycle_type)


def pmg_afw(formdegree, mg_type, pcoarse, factor=None, steps=1, cycles=1, mat_type=None, pmat_type=None, **kwargs):
    if factor is None:
        factor = dict()
    if mat_type is None:
        mat_type = "matfree"

    aux = lambda x: x
    if not (mat_type.endswith("aij") or factor.get("pc_type", "").endswith("jacobi")):
        aux = lambda x: fdm({**x, "mat_type": pmat_type or "aij", "static_condensation": True,})

    plevels = star_relax(0, factor)
    if mg_type != "additive":
        plevels.update(cheb(steps, **kwargs))

    return aux(pmg(pcoarse, plevels, mg_type=mg_type, cycles=cycles))


def pmg_hiptmair(formdegree, mg_type, pcoarse, factor=None,
                 factor_potential=None, steps=1, cycles=1, mat_type=None, pmat_type=None, **kwargs):

    if mat_type is None:
        mat_type = "matfree"
    aux = lambda x: x
    if not mat_type.endswith("aij"):
        aux = lambda x: fdm({**x, "mat_type": pmat_type or "aij", "static_condensation": True,})

    relax = star_relax(formdegree, factor)
    relax_potential = star_relax(formdegree-1, factor_potential)
    plevels = hiptmair(aux(relax_potential), relax, mg_type="additive")
    if mg_type != "additive":
        plevels.update(cheb(steps, **kwargs))
    return aux(pmg(pcoarse, plevels, mg_type=mg_type, cycles=cycles))


def pmg_facet_hiptmair(formdegree, mg_type, pcoarse, factor=None,
                       factor_potential=None, steps=1, cycles=1, mat_type=None, pmat_type=None, **kwargs):
    aux = lambda x: x
    if mat_type != "aij":
        aux = lambda x: fdm({**x, "mat_type": pmat_type or "aij", "static_condensation": True,})
    relax = star_relax(formdegree, factor)
    relax_potential = star_relax(formdegree-1, factor_potential)
    facet_pc = hiptmair(aux(relax_potential), aux(relax), mg_type="additive")
    interior_pc = bjacobi(icc(0)) if formdegree else jacobi()
    facetsplit = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.FacetSplitPC",
        "facet": schur(interior_pc, facet_pc),
    }
    plevels = facetsplit
    if mg_type != "additive":
        plevels.update(cheb(steps, **kwargs))
    return aux(pmg(pcoarse, plevels, mg_type=mg_type, cycles=cycles))


def pmg_facet_afw(formdegree, mg_type, pcoarse, factor=None, steps=1, cycles=1, mat_type=None, pmat_type=None, **kwargs):
    aux = lambda x: x
    if mat_type != "aij":
        aux = lambda x: fdm({**x, "mat_type": pmat_type or "aij", "static_condensation": True,})
    facet_pc = aux(star_relax(0, factor))
    interior_pc = bjacobi(icc(0)) if formdegree else jacobi()
    facetsplit = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.FacetSplitPC",
        "facet": schur(interior_pc, facet_pc),
    }
    plevels = facetsplit
    if mg_type != "additive":
        plevels.update(cheb(steps, **kwargs))
    return aux(pmg(pcoarse, plevels, mg_type=mg_type, cycles=cycles))


def pmg_jacobi(coarse, steps=1, scale=0.125):
    levels = jacobi()
    levels.update(cheb(steps, scale=scale))
    return {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.PMGPC",
        "pmg_coarse_degree": 1,
        "pmg_coarse_mat_type": "aij",
        "pmg_mg_coarse": coarse,
        "pmg_mg_levels": levels,
        "pmg_pc_mg_log": mg_log,
    }


def variable_v_cycle(sparams, prefix, steps, nlevels):
    for l in range(1, nlevels):
        sparams["%smg_levels_%d_ksp_max_it" % (prefix, l)] = steps*(2**(nlevels-l-1))
    return sparams


def facetsplit_redundant(V, facet_pc, pmat_type=None, appctx=None):
    if appctx is None:
        appctx = dict()
    formdegree = V.finat_element.formdegree

    aux = lambda x: fdm({**x, "mat_type": pmat_type or "aij","static_condensation": True})
    interior_pc = bjacobi(icc(0)) if formdegree else (pbjacobi() if V.shape else jacobi())
    interior_pc = aux(interior_pc)
    facet_pc["ksp_type"] = "preonly"
    sparams = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.FacetSplitPC",
        "facet": schur(interior_pc, facet_pc),
    }
    return aux(sparams)


def facetsplit_nest(V, facet_pc, pmat_type=None, appctx=None):
    if appctx is None:
        appctx = dict()
    formdegree = V.finat_element.formdegree
    aux = lambda x: fdm({**x, "mat_type": pmat_type or "aij", "static_condensation": True, "pc_use_amat": False})
    interior_pc = bjacobi(icc(0)) if formdegree else (pbjacobi() if V.shape else jacobi())
    if "fdm" in facet_pc:
        facet_pc = facet_pc["fdm"]
    facet_pc["ksp_type"] = "preonly"
    sparams = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.FacetSplitPC",
        "facet": aux(bdiag(interior_pc, facet_pc, fieldsplit_type="symmetric_multiplicative")),
    }
    return sparams


def facetsplit_matfree(V, facet_pc, pmat_type=None, appctx=None):
    if appctx is None:
        appctx = dict()
    formdegree = V.finat_element.formdegree
    interior_pc = bjacobi(icc(0)) if formdegree else (pbjacobi() if V.shape else jacobi())

    steps = 8
    use_cheb = steps > 1

    if use_cheb:
        aux = lambda x: x
        facet_pc["fdm_static_condensation"] = True
        facet_pc["fdm_pc_use_amat"] = False
        facet_pc["fdm_pmg_pc_use_amat"] = False

        if formdegree:
            steps = 21
            interior_pc = fdm({**interior_pc, "pc_use_amat": False, "mat_type": pmat_type or "aij"})
            interior_pc["ksp_diagonal_scale"] = dscale
            interior_pc["ksp_diagonal_scale_fix"] = dscale
        interior_pc.update(cheb(steps, scale=0))

        #interior_pc.pop("ksp_chebyshev_esteig", None)
        #interior_pc["ksp_chebyshev_eigenvalues"] = "0.5,1.5"

    elif "fdm" in facet_pc:
        mat_type = facet_pc["fdm"]["mat_type"]
        facet_pc = facet_pc["fdm"]
        aux = lambda x: fdm({**x, "pc_use_amat": False, "static_condensation": True, "mat_type": mat_type})

    facet_pc["ksp_type"] = "preonly"
    sparams = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.FacetSplitPC",
        "facet": aux(bdiag(interior_pc, facet_pc, fieldsplit_type="symmetric_multiplicative")),
        "facet_mat_type": "matfree" if use_cheb else "submatrix",
    }
    return sparams


def update_appctx(V, appctx):
    degree = V.finat_element.degree
    if isinstance(degree, tuple):
        degree = max(degree)
    formdegree = V.finat_element.formdegree
    if formdegree > 0:
        appctx["get_gradient"] = tabulate_exterior_derivative
    if formdegree > 1:
        appctx["get_curl"] = tabulate_exterior_derivative
        if degree > 1:
            appctx["hiptmair_shift"] = firedrake.Constant(1E-8)


def pc_jacobi(bsize):
    return jacobi() if bsize == 1 else pbjacobi()


def pc_gmg(V, coarse, cycles=1, steps=1, scale=0.125, mg_type="multiplicative", **kwargs):
    _, level = get_level(V.mesh())
    if level == 0:
        return coarse
    formdegree = V.finat_element.formdegree
    value_size = V.value_size
    levels = pc_jacobi(value_size)
    if formdegree:
        levels = hiptmair(levels, levels)
    if mg_type != "additive":
        levels.update(cheb(steps, scale=scale))
    return gmg(assembled(coarse), levels, mg_type=mg_type)


def pc_coarse(V, appctx=None, coarse_solver=None, **kwargs):
    if appctx is not None:
        update_appctx(V, appctx)
    if coarse_solver == "hypre":
        formdegree = V.finat_element.formdegree
        coarse = [hypre, hypre_ams, hypre_ads][formdegree]
    elif coarse_solver == "lu":
        coarse = lu(solver="mumps")
    else:
        coarse = chol(solver="mumps")
    return pc_gmg(V, coarse, **kwargs)


def pc_factor(formdegree, value_size, restriction=None, pmat_type="aij"):
    symmetric = pmat_type.endswith("sbaij")
    if restriction == "interior":
        if formdegree == 0:
            return pc_jacobi(value_size)
        else:
            # FIXME inplace ilu(0)
            return bjacobi(icc(0) if symmetric else ilu(0, in_place=False))

    elif restriction == "facet" and formdegree == 2:
        return pc_jacobi(value_size)

    elif formdegree == 0:
        return icc(0) if symmetric else ilu(0)

    return chol(solver="cholmod") if symmetric else lu(solver="umfpack")


def pc_asm(formdegree, value_size, restriction=None, pmat_type="aij"):
    factor = pc_factor(formdegree, value_size, restriction=restriction, pmat_type=pmat_type)
    if factor.get("pc_type", "").endswith("jacobi"):
        return factor
    return star(formdegree, factor)


def pc_hiptmair_asm(formdegree, value_size, restriction=None, pmat_type="aij", variant=None):
    levels = pc_asm(formdegree, value_size, restriction=restriction, pmat_type=pmat_type)
    if formdegree == 0:
        return levels
    else:
        aux = fdm if variant == "fdm" else assembled
        coarse = pc_asm(formdegree-1, value_size, restriction=restriction, pmat_type=pmat_type)

        use_sc = restriction == "facet"
        coarse = aux(coarse, mat_type=pmat_type, pc_use_amat=not use_sc, static_condensation=use_sc)
        return hiptmair(coarse, levels)


def pc_afw_asm(formdegree, value_size, restriction=None, pmat_type="aij", variant=None):
    symmetric = pmat_type.endswith("sbaij")
    factor = chol(solver="cholmod", ordering_type="nd") if symmetric else lu(solver="umfpack")
    return star(0, factor)


def pc_pmg(coarse, levels, cycles=1, steps=1, scale=0.125, mg_type="multiplicative", **kwargs):
    if mg_type == "none":
        return levels
    if mg_type != "additive":
        levels.update(cheb(steps, scale=scale))
    return pmg(coarse, levels, mg_type=mg_type)


def pc_pmg_fdm(V, appctx=None, static_condensation=False, pmat_type="aij", relax="afw", **kwargs):
    """
    Returns one out of four solvers P-AFW, P-AFW/SC, PH, PH/SC
    Composes p-MG as the outermost solver with teh sparse FDM as the relaxation
    """
    degree = V.ufl_element().degree()
    if isinstance(degree, tuple):
        degree = max(degree)
    cycles = None
    if degree > 1:
        cycles = kwargs.pop("cycles", cycles)
    coarse = pc_coarse(V, appctx=appctx, **kwargs)
    if degree == 1:
        return coarse
    if cycles is not None:
        kwargs["cycles"] = cycles

    if relax == "jacobi":
        pc_space_decomp = lambda fd, vs, **kwargs: pc_jacobi(vs)
    elif relax == "afw":
        pc_space_decomp = pc_afw_asm
    else:
        pc_space_decomp = pc_hiptmair_asm

    formdegree = V.finat_element.formdegree
    value_size = V.value_size
    variant = V.ufl_element().variant()
    aux = fdm if variant == "fdm" else assembled
    if static_condensation:
        assert variant == "fdm"
        interior = pc_factor(formdegree, value_size, restriction="interior", pmat_type=pmat_type)
        facet = pc_space_decomp(formdegree, value_size, restriction="facet", pmat_type=pmat_type, variant=variant)
        patches = bdiag(interior, facet, fieldsplit_type="symmetric_multiplicative")
        levels = facetsplit(aux(patches, mat_type=pmat_type, pc_use_amat=False, static_condensation=True))
    else:
        patches = pc_space_decomp(formdegree, value_size, pmat_type=pmat_type)
        levels = aux(patches, mat_type=pmat_type)
    return pc_pmg(coarse, levels, **kwargs)


def pc_fdm_pmg(V, appctx=None, static_condensation=False, pmat_type="aij", relax="afw", **kwargs):
    """
    Returns one out of four solvers P-AFW, P-AFW/SC, PH, PH/SC
    Composes the sparse FDM relaxation as the outermost solver with p-MG as the inner solver
    """
    degree = V.ufl_element().degree()
    if isinstance(degree, tuple):
        degree = max(degree)
    cycles = None
    if degree > 1:
        cycles = kwargs.pop("cycles", cycles)
    coarse = pc_coarse(V, appctx=appctx, **kwargs)
    if degree == 1:
        return coarse
    if cycles is not None:
        kwargs["cycles"] = cycles

    if relax == "jacobi":
        pc_space_decomp = lambda fd, vs, **kwargs: pc_jacobi(vs)
    elif relax == "afw":
        pc_space_decomp = pc_afw_asm
    else:
        pc_space_decomp = pc_hiptmair_asm

    formdegree = V.finat_element.formdegree
    value_size = V.value_size
    variant = V.ufl_element().variant()
    aux = fdm if variant == "fdm" else assembled
    if static_condensation:
        assert variant == "fdm"
        interior = pc_factor(formdegree, value_size, restriction="interior", pmat_type=pmat_type)
        facet = pc_space_decomp(formdegree, value_size, restriction="facet", pmat_type=pmat_type, variant=variant)
        facet = pc_pmg(coarse, facet, **kwargs)
        patches = bdiag(interior, facet, fieldsplit_type="symmetric_multiplicative")
        return facetsplit(aux(patches, mat_type=pmat_type, pc_use_amat=False, static_condensation=True))
    else:
        levels = pc_space_decomp(formdegree, value_size, pmat_type=pmat_type)
        patches = pc_pmg(coarse, levels, **kwargs)
        return aux(patches, mat_type=pmat_type)


def pc_mardal_winther(Z, appctx=None,
                      riesz_ksp_type="chebyshev",
                      mass_ksp_type="chebyshev",
                      static_condensation=False, pmat_type="aij", **kwargs):
    """
    Returns optimal-complexity solver parameters for a given function space
    """
    riesz_ksp = {}
    steps = 0
    rtol = 1E-3
    atol = 0E-12
    if static_condensation and len(Z) > 1:
        steps = kwargs.pop("steps", 1)
        if steps > 1:
            if riesz_ksp_type == "chebyshev":
                riesz_ksp = cheb(steps, **kwargs)
            elif riesz_ksp_type != "preonly":
                norm_type = "natural" if riesz_ksp_type == "cg" else "unpreconditioned"
                pc_side = "right" if norm_type == "unpreconditioned" else "left"
                riesz_ksp = {"ksp_type": riesz_ksp_type,
                             "ksp_norm_type": norm_type,
                             "ksp_pc_side": pc_side,
                             "ksp_max_it": steps,
                             "ksp_rtol": rtol,
                             "ksp_atol": atol,
                             "ksp_monitor": None,
                            }
    mass_ksp = {}
    steps = kwargs.get("steps", max(1, steps))
    if len(Z) > 1 and steps > 1:
        if mass_ksp_type == "chebyshev":
            mass_ksp = cheb(steps,
                            esteig_ksp_type="gmres",
                            esteig_ksp_norm_type="preconditioned",
                            **kwargs)
        elif mass_ksp_type not in ["preonly", "none"]:
            norm_type = "natural" if mass_ksp_type == "cg" else "unpreconditioned"
            pc_side = "right" if norm_type == "unpreconditioned" else "left"
            mass_ksp = {"ksp_type": mass_ksp_type,
                        "ksp_pc_side": pc_side,
                        "ksp_norm_type": norm_type,
                        "ksp_max_it": steps,
                        "ksp_rtol": rtol,
                        "ksp_atol": atol,
                        "ksp_monitor": None,
                       }

    tdim = Z.mesh().topological_dimension()
    pcs = []
    for V in Z:
        formdegree = V.finat_element.formdegree
        if formdegree == tdim:
            if mass_ksp_type == "none":
                pc = pcnone()
            else:
                pc = pc_jacobi(V.value_size)
                pc.update(mass_ksp)
        else:
            pc = pc_fdm_pmg(V, appctx=appctx, pmat_type=pmat_type,
                            static_condensation=static_condensation,
                            **kwargs)
            pc.update(riesz_ksp)
        pcs.append(pc)

    return bdiag(*pcs) if len(pcs) > 1 else pcs[0]
