import mpi4py.MPI as MPI

import ngsolve as ngs
import netgen as ng
import netgen.geom2d as geom2d
import netgen.csg as csg
import ngsolve.krylovspace
from netgen.geom2d import unit_square
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh
from ngsolve import *
from netgen.occ import *
from ngsPETSc import KrylovSolver, NullSpace, PETScPreconditioner

def petscGAMGH1Precond(a):
    V = a.space

    tSup = Timer("PETSc - setup")

    MPI.COMM_WORLD.Barrier()
    tSup.Start()

    pc = PETScPreconditioner(a.mat, V.FreeDofs(), solverParameters={"pc_type": "gamg"})

    MPI.COMM_WORLD.Barrier()
    tSup.Stop()

    if MPI.COMM_WORLD.rank == 0:
        print(f"PETSc GAMG setup = {tSup.time}")

    return pc


def petscGAMGElasticityPrecond(a):
    V = a.space

    dim = V.mesh.ngmesh.dim

    if dim == 3:
        rb_funcs = [(1,0,0), (0,1,0), (0,0,1), (-y,x,0), (0,x,-z), (z,0,-x)]
    else:
        rb_funcs = [(1,0),(0,1),(-y,x)]    

    rbms = []
    for val in rb_funcs:
        rbm = GridFunction(V)
        rbm.Set(CF(val))
        rbms.append(rbm.vec)

    nullspace = NullSpace(V, rbms, near=True)

    tSup = Timer("PETSc - setup")

    MPI.COMM_WORLD.Barrier()
    tSup.Start()

    pc = PETScPreconditioner(a.mat, V.FreeDofs(), nullspace=nullspace,
         solverParameters={"pc_type": "gamg"})
    
    MPI.COMM_WORLD.Barrier()
    tSup.Stop()

    if MPI.COMM_WORLD.rank == 0:
        print(f"PETSc GAMG setup = {tSup.time}")

    return pc