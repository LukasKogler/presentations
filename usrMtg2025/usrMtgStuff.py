import ngsolve as ngs
import netgen as ng
import netgen.geom2d as geom2d
import netgen.csg as csg
import ngsolve.krylovspace
from netgen.geom2d import unit_square
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh
from ngsolve import *
from netgen.occ import *


def gen_ref_mesh (geo, maxh, nref, comm, mesh_file = '', save = False, load = False):
    ngs.ngsglobals.msg_level = 1
    if load:
        ngm = ng.meshing.Mesh(comm=comm)
        ngm.Load(mesh_file)
    else:
        if comm.rank==0:
            ngm = geo.GenerateMesh(maxh=maxh)
            if save:
                ngm.Save(mesh_file)
            if comm.size > 0:
                ngm.Distribute(comm)
        else:
            ngm = ng.meshing.Mesh.Receive(comm)
            ngm.SetGeometry(geo)
    for l in range(nref):
        ngm.Refine()
    return geo, ngs.comp.Mesh(ngm)


def setupH1Square(maxh=0.1, order=1, nref=0):
    ngmesh = unit_square.GenerateMesh(maxh=maxh)
    for l in range(nref):
        ngmesh.Refine()
    mesh = ngs.Mesh(ngmesh)
    V = ngs.H1(mesh, order=order, dirichlet=".*")
    u,v = V.TnT()
    a = ngs.BilinearForm(V)
    a += ngs.Grad(u) * ngs.Grad(v) * ngs.dx
    f = ngs.LinearForm(V)
    f += v * ngs.dx
    gfu = ngs.GridFunction(V)
    return V, a, f, gfu


def setupElastBeam(N=3, aRatio=10, pRatio = 0.1, elStretch=1, order=1, nodalP2=False, dim=3):
    if aRatio%elStretch != 0:
        raise Exception(f"aRatio={aRatio} is not a multiple of elStretch={elStretch}!")
    nX = N * aRatio // elStretch
    if dim == 3:
        mesh = MakeStructured3DMesh(hexes=False, nx=nX, ny=N, nz=N, secondorder=False,
                                    mapping = lambda x,y,z : (x,y/N,z/N))
        diri="back"
    else:
        mesh = MakeStructured2DMesh(quads=False, nx=nX, ny=N, secondorder=False,
                                    mapping = lambda x,y : (x,y/N))
        diri="left"

    extraOpts = {}
    if nodalP2 and order > 1:
        extraOpts["nodalp2"] = True
    V = ngs.H1(mesh, dim=dim, order=order, dirichlet=diri, **extraOpts)
    u,v = V.TnT()

    # convert Young's modulus E and poisson's ratio nu to lame parameters mu,lambda
    E = 1.0
    mu  = E / ( 2 * ( 1 + pRatio ))                                # eps-eps
    lam = ( E * pRatio ) / ( (1 + pRatio) * (1 - (2 * pRatio) ) )  # div-div

    sym = lambda X : 0.5 * (X + X.trans)
    grd = lambda X : ngs.CoefficientFunction( tuple(ngs.grad(X)[i,j] for i in range(dim) for j in range(dim)), dims=(dim,dim))
    eps = lambda X : sym(grd(X))
    div = lambda U : sum([ngs.grad(U)[i,i] for i in range(1, dim)], start=ngs.grad(U)[0,0])

    a = ngs.BilinearForm(V, symmetric=False)
    a += mu * ngs.InnerProduct(eps(u), eps(v)) * ngs.dx
    a += lam * div(u) * div(v) * ngs.dx

    rhsCFs=[None, None, (0, 1e-3 * ngs.x), (0, 1e-3 * ngs.x, 0) ]

    f = ngs.LinearForm(V)
    f += ngs.InnerProduct(ngs.CF(rhsCFs[dim]), v) * ngs.dx

    gfu = ngs.GridFunction(V)
    return V, a, f, gfu


def testAndSolve(A, c, u, f, doTest=True, tol=1e-6):
    evsA = ngs.la.EigenValues_Preconditioner(mat=A, pre=c)
    kappa = evsA[-1]/evsA[0]
    print("Preconditioner test:")
    print(f"   min EV = {evsA[0]}")
    print(f"   max EV = {evsA[-1]}")
    print(f"   condition = {kappa}")
    print("")
    print("")

    sol = ngs.krylovspace.CGSolver(mat=A, pre=c, tol=tol, printrates=True)

    print("Solve...")
    t = ngs.Timer("AMG - Solve")
    t.Start()
    u.data = sol * f
    t.Stop()

    tsup = -1
    for timer in ngs.Timers():
        if timer["name"] == "BaseAMGPC::BuildAMGMat":
            # "Matrix assembling finalize matrix" would also work
            tsup = timer["time"]

    print("")
    print(f"time AMG setup = {tsup} sec")
    print(f"    set up  {A.height / tsup} DOFS/sec")
    print(f"time solve = {t.time} sec")
    print(f"    solved  {A.height / t.time} DOFS/sec")

    return kappa

def solveCondensed(a, c, u, f, tol=1e-6):

    hex  = a.harmonic_extension
    hexT = a.harmonic_extension_trans
    aii  = a.inner_matrix
    Id = ngs.IdentityMatrix(a.mat.height)
    A = (Id - hexT) @ (a.mat + aii) @ (Id - hex)

    # residual form; solve Ax = (b-Au)
    b = f.CreateVector()
    b.data = f - A * u
    x = u.CreateVector()
    x.data[:] = 0

    # hexT
    b.data += hexT * b
    # solve SC
    testAndSolve(a.mat, c, x, b, tol=tol)
    # hex
    x.data += a.inner_solve * b
    x.data += hex * x
    u.data += x



def GetValve(N=1, dim=2, R=0.5, alpha=25, Winlet=1, beta=180, L1=6.4, L2=7, Linlet=5, closevalve=False, maxh=0.5):
    alphar = alpha * pi / 180
    W = 1 / cos(alphar / 2)
    facW = 1
    Winlet2 = Winlet
    wp = WorkPlane()
    p1 = 0
    p2 = 0
    wp.MoveTo(p1, p1)
    r1 = Rectangle(L1, W).Face()
    r2 = wp.MoveTo(p1, p2 + W).Rotate(-90 + alpha).Move(W).Rotate(90).Rectangle(L2, W).Face()
    wp.MoveTo(p1, p2 + W).Move(L2)
    c1 = Face(wp.Arc(W + R, -beta).Line(L1).Rotate(-90).Line(W).Rotate(-90).Line(L1).Arc(R, beta).Close().Wire())
    wp.MoveTo(0, W).Direction(1, 0)
    cutplane = Rectangle(2 * L1, 4 * L1).Face()
    v1 = r1 + r2 + cutplane * c1
    ll = L1 + L1 * cos(alphar) - cos(alphar) * W
    hh = L1 * sin(alphar) - (1 - sin(alphar)) * W
    didi = sqrt((L1 + L1 * cos(alphar)) ** 2 + (L1 * sin(alphar)) ** 2) - 2 * W * sin(alphar / 2)
    dd = gp_Dir(cos(alpha), sin(alpha), 0)
    v2 = v1.Mirror(Axis((0, 0, 0), X)).Move((0, W, 0)).Rotate(Axis((0, 0, 0), Z), alpha).Move((L1, 0, 0))
    onevalve = (v1 + v2.Reversed()).Move((0, -W / 2, 0)).Rotate(Axis((0, 0, 0), Z), -alpha / 2)
    vv = onevalve
    for i in range(1, N):
        vv = onevalve.Move((didi * i, 0, 0)) + vv
    inlet = wp.MoveTo(-Linlet, -Winlet2 / 2).Rectangle(Linlet * facW + W * sin(alphar / 2), Winlet2).Face()
    outlet = wp.MoveTo(didi * N + (W / 2) * sin(alphar / 2) * facW, -Winlet2 / 2).Rectangle(Linlet, Winlet2).Face()
    vv = inlet + vv + outlet
    if closevalve == True:
        c1_x = -Linlet
        c1_y = -Winlet2 / 2
        c2_x = didi * N + (W / 2) * sin(alphar / 2) * facW + Linlet
        c2_y = -Winlet2 / 2
        wp.MoveTo(c1_x, c1_y).Direction(-1, 0)
        R2 = 3
        LL = c2_x - c1_x
        close = Face(wp.Arc(R2 + Winlet2, -180).Line(LL).Arc(R2 + Winlet2, -180).Rotate(-90).Line(Winlet2).Rotate(-90).Arc(R2, 180).Line(LL).Arc(R2, 180).Close().Wire()).Reversed()
    teslavalve = vv
    if dim == 3:
        teslavalve = vv.Extrude(Winlet * Z)
        teslavalve.faces.name = 'wall'
        teslavalve.solids.name = 'valve'
        teslavalve.faces.Min(X).name = 'inlet'
        teslavalve.faces.Max(X).name = 'outlet'
        if closevalve == True:
            close = close.Extrude(Winlet * Z)
            teslavalve = Glue([
                teslavalve,
                close])
        mesh = Mesh(netgen.occ.OCCGeometry(teslavalve, dim=dim).GenerateMesh(maxh=maxh))
        return mesh
        # return teslavalve
    teslavalve.edges.name = None
    teslavalve.faces.name = 'valve'
    teslavalve.edges.Min(X).name = 'inlet'
    teslavalve.edges.Max(X).name = 'outlet'
    if closevalve == True:
        close.edges.name = 'wall'
        teslavalve = Glue([
            teslavalve,
            close])
        
    # return teslavalve
    mesh = Mesh(netgen.occ.OCCGeometry(teslavalve, dim=dim).GenerateMesh(maxh=maxh))
    # mesh.Curve(3)
    return mesh

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