from ngsolve import *
from ngsolve.webgui import Draw
import ngsolve as ngs
import NgsAMG as amg
import mpi4py.MPI as MPI

from netgen.csg import unit_cube
from usrMtgStuff import StokesHDGDiscretization

def gen_ref_mesh (geo, maxh, nref, comm):
    ngs.ngsglobals.msg_level = 1
    if comm.rank==0:
        ngm = geo.GenerateMesh(maxh=maxh)
        if comm.size > 0:
            ngm.Distribute(comm)
    else:
        ngm = ng.meshing.Mesh.Receive(comm)
    ngm.SetGeometry(geo)
    for l in range(nref):
        ngm.Refine()
    return geo, ngs.comp.Mesh(ngm)

geo,mesh = gen_ref_mesh(unit_cube, maxh=0.2, nref=1, comm=MPI.COMM_WORLD)

print(mesh.ne)