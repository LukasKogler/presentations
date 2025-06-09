from ngsolve import *
from ngsolve.webgui import Draw
import ngsolve as ngs
import NgsAMG as amg

from usrMtgStuff import GetValve, testAndSolve, solveCondensed



def StokesHDGDiscretizationTwoSpace(mesh, order, inlet, wall, div_div_pen, outlet, hodivfree=True, proj_jumps=True, nu=1e-4, elint=True, f=None):
    diri=f"{wall}|{inlet}"
    V1 = HDiv(mesh, order=order, dirichlet=diri, hodivfree=hodivfree, RT=False)
    V2 = TangentialFacetFESpace(mesh, order=order, dirichlet=diri, highest_order_dc=proj_jumps)
    V = V1 * V2
    Q = L2(mesh, order = 0 if hodivfree else order - 1)
    (u, uhat), (v, vhat) = V.TnT()
    (p, q) = Q.TnT()
    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    
    def tang(vec = None):
        return vec - vec * n * n

    alpha = 4
    dS = dx(element_boundary=True)
    a = BilinearForm(V, eliminate_internal=elint, store_inner=True)
    a += nu * InnerProduct(Grad(u), Grad(v)) * dx
    a += nu * InnerProduct(Grad(u) * n, tang(vhat - v)) * dS
    a += nu * InnerProduct(Grad(v) * n, tang(uhat - u)) * dS
    a += (nu * alpha * order * order / h) * InnerProduct(tang(vhat - v), tang(uhat - u)) * dS
    if div_div_pen is not None:
        aPen = BilinearForm(V, eliminate_internal=elint, store_inner=True)
        aPen += nu * InnerProduct(Grad(u), Grad(v)) * dx
        aPen += nu * InnerProduct(Grad(u) * n, tang(vhat - v)) * dS
        aPen += nu * InnerProduct(Grad(v) * n, tang(uhat - u)) * dS
        aPen += (nu * alpha * order * order / h) * InnerProduct(tang(vhat - v), tang(uhat - u)) * dS
        aPen += div_div_pen * nu * InnerProduct(div(u), div(v)) * dx
    else:
        aPen = a
    b = BilinearForm(trialspace=V, testspace=Q)
    b += -div(u) * q * dx

    lf = LinearForm(V)
    if f is not None:
        lf += ngs.InnerProduct(f, v) * ngs.dx

    lg = LinearForm(Q)

    return (V, Q, a, b, aPen, lf, lg)

def StokesHDGDiscretization(mesh, order, inlet, wall, div_div_pen, outlet, hodivfree=True, proj_jumps=True, nu=1e-4, elint=True, f=None, beta=0):
    diri=f"{wall}|{inlet}"
    V1 = HDiv(mesh, order=order, dirichlet=diri, hodivfree=hodivfree, RT=False)
    V2 = TangentialFacetFESpace(mesh, order=order, dirichlet=f"{diri}|{outlet}", highest_order_dc=proj_jumps)
    V = V1 * V2
    (u, uhat), (v, vhat) = V.TnT()
    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    
    def tang(vec = None):
        return vec - vec * n * n

    alpha = 4
    dS = dx(element_boundary=True)

    aPen = BilinearForm(V, eliminate_internal=elint, store_inner=True)
    aPen += alpha * nu * InnerProduct(Grad(u), Grad(v)) * dx
    aPen += nu * InnerProduct(Grad(u) * n, tang(vhat - v)) * dS
    aPen += nu * InnerProduct(Grad(v) * n, tang(uhat - u)) * dS
    aPen += (nu * alpha * order * order / h) * InnerProduct(tang(vhat - v), tang(uhat - u)) * dS
    aPen += div_div_pen * nu * InnerProduct(div(u), div(v)) * dx

    # aPen += 1e3 * nu * InnerProduct(uhat.Trace(), vhat.Trace()) * ngs.ds(definedon=mesh.Boundaries("outlet"))

    if beta > 0:
        aPen += beta * InnerProduct(u, v) * dx

    # aPen += 1e3 * nu * alpha * InnerProduct(Grad(u), Grad(v)) * dx

    lf = LinearForm(V)
    if f is not None:
        lf += ngs.InnerProduct(f, v) * ngs.dx

    u = GridFunction(V)

    return (V, aPen, lf, u)


# valve = GetValve(N=1, dim=dim, R=0.5, alpha=25, Winlet=1, beta=180, L1=6.4, L2=7, Linlet=5, closevalve=True)
# mesh = Mesh(netgen.occ.OCCGeometry(valve, dim=dim).GenerateMesh(maxh=0.5))
# mesh.Curve(3)
# mesh = GetValve(N=5, dim=dim, R=0.5, alpha=25, Winlet=1, beta=180, L1=6.4, L2=7, Linlet=5, closevalve=False, maxh=1.0)
# mesh = GetValve(N=5, dim=2, maxh=10, closevalve=False)
# mesh.Curve(3)
mesh = GetValve(N=4, dim=2, maxh=0.5, closevalve=False)

uin = CF((1, 0)) if mesh.ngmesh.dim == 2 else CF((1,0,0))
# f_vol = CF((1, 0)) if dim == 2 else CF((1,0,0))
f_vol = None

# (V, a, f, u) = StokesHDGDiscretization(mesh, order=2, wall="wall", inlet="inlet", outlet="", nu=1e-3, div_div_pen=0, f=f_vol)
wall=""
inlet="inlet"
beta=1e-5 # no diri, no beta -> singular
wall = "wall" if mesh.ngmesh.dim == 3 else "default"
(V, a, f, u) = StokesHDGDiscretization(mesh, order=2, wall=wall, inlet=inlet, outlet="", nu=1e-3, beta=beta, div_div_pen=1e5, f=f_vol)

print("# Elements = ", mesh.ne)
print("# DOFs = ", V.components[0].ndof, V.components[1].ndof)
# quit()

amg_cl = amg.stokes_hdiv_gg_2d if mesh.ngmesh.dim == 2 else amg.stokes_hdiv_gg_3d

pc_opts = {
    'ngs_amg_max_levels': 30,
    'ngs_amg_max_coarse_size': 1,
    'ngs_amg_clev': ['none','inv'][1],
    "ngs_amg_cinv_type_loc": "sparsecholesky",
    'ngs_amg_log_level': ["extra", "debug"][0],
    'ngs_amg_log_level_pc': ["extra", "debug"][0],
    'ngs_amg_do_test': True,
    "ngs_amg_crs_alg": "spw",
    "ngs_amg_test_smoothers": False,
    "ngs_amg_test_levels": False,
    "ngs_amg_pres_vecs": ["RTZ", "P0", "P1"][2],
    "ngs_amg_sm_steps_spec" : [ 1 ],
    "ngs_amg_mg_cycle" : ["BS", "V"][1],
    "ngs_amg_sm_type"  : ["hiptmair", "amg_smoother", "gs", "bgs"][0],
    "ngs_amg_sm_type_pot"   : "gs",
    "ngs_amg_sm_type_range" : ["gs", "bgs", "dyn_block_gs"][2],
    "ngs_amg_sm_type_range_spec" : [],#"dyn_block_gs"],#[ "bgs" ][:0 if disc_opts["order"] == 1 else 1],
    "ngs_amg_sm_type_spec": [],
    "ngs_amg_use_dynbs_prols": True,
}

# c = amg_cl(a, **pc_opts)
c = amg_cl(a, ngs_amg_pres_vecs="P1", ngs_amg_sm_type="hiptmair", ngs_amg_sm_type_range="dyn_block_gs")
# c = amg_cl(a, ngs_amg_pres_vecs="P1", ngs_amg_sm_type="hiptmair", ngs_amg_sm_type_range="dyn_block_bgs", ngs_amg_do_test=True)

a.Assemble()

u.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

solveCondensed(a, c, u.vec, f.vec, tol=1e-8)
quit()


f.Assemble()

r = f.vec.CreateVector()
# sol = f.vec.CreateVector()
u.components[0].Set(uin)

r.data = f.vec - a.mat * u.vec

# for t in Timers():
#     if t["time"] > 0.1:
#         print(t["name"], t["time"])
x = f.vec.CreateVector()
x.data = u.vec

testAndSolve(a, c, x, r, tol=1e-8)

u.vec.data += x

for t in Timers():
    if t["time"] > 0.1:
        print(t["name"], t["time"])