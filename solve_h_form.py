#%%
import numpy as np
from basix.ufl import element
from dolfinx import fem
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import (
    Function,
    dirichletbc,
    form,
    locate_dofs_topological,
    petsc,
)
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import GhostMode
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    curl,
    inner,
    variable,
    sin,
    pi,
    sqrt,
)
from dolfinx import default_scalar_type
from utils import L2_norm, par_print, interpolate_by_tags

with XDMFFile(MPI.COMM_WORLD, "rod.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(ghost_mode=GhostMode.none)
    ct = xdmf.read_meshtags(domain, name="ct")
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, 0)
    ft = xdmf.read_meshtags(domain, name="ft")
    material_tags = np.unique(ct.values)
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

boundary_tags = {
    "cube_boundary": 1,
    "upper_surface": 2,
    "side_surface": 3,
    "bottom_surface": 4,
}

vol_ids = {"copper": 1, "air": 2}

const = fem.functionspace(domain, ("DG", 0))  # Piecewise constant function space

mu = 4e-7 * np.pi

Ec = 1e-4
Jc = 1e5
rho_air = 1.0
# Linear (constant) resistivity in the conductor: the n = 1 limit of the
# power law E = Ec (J/Jc)^n, i.e. rho = Ec / Jc.
rho_cond = Ec / Jc

# Piecewise-constant linear resistivity, one value per material tag.
# Independent of H, so the problem is linear.
rho = fem.Function(const)
interpolate_by_tags(
    rho,
    {vol_ids["copper"]: rho_cond, vol_ids["air"]: rho_air},
    ct,
)

comm = MPI.COMM_WORLD
degree = 1

V_CG = fem.functionspace(domain, ("CG", degree))

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 100  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
H_space = fem.functionspace(domain, nedelec_elem)


x = SpatialCoordinate(domain)
h_n = fem.Function(H_space)

H = TrialFunction(H_space)
v = TestFunction(H_space)

dx = Measure("dx", domain, subdomain_data=ct, metadata={"quadrature_degree": 2})


# Linear H-formulation: mu dH/dt + curl(rho curl H) = 0 (backward Euler)
lhs = (mu / dt) * inner(H, v) * dx + inner(rho * curl(H), curl(v)) * dx
rhs = (mu / dt) * inner(h_n, v) * dx

a = form(lhs)
L = form(rhs)

# Boundary conditions

boundary_tags_V = (boundary_tags["upper_surface"],boundary_tags["cube_boundary"],boundary_tags["bottom_surface"])
boundary_facets_V = np.concatenate([ft.find(tag) for tag in boundary_tags_V])


B0 = 0.01  # Applied flux density magnitude in T
# Uniform, time-independent applied field along z
direction = np.array([0.0, 0.0, 1.0])

def H_applied(t_val):
    amp = B0 / mu
    return lambda x: np.vstack(
        [amp * direction[i] * np.ones_like(x[0]) for i in range(3)]
    )

dofs = locate_dofs_topological(V=H_space, entity_dim=fdim, entities=boundary_facets_V)
H_bc = Function(H_space)
H_bc.interpolate(H_applied(ti))
bc = dirichletbc(H_bc, dofs)

# Solver steps. The operator is time independent for the linear problem,
# so it is assembled once.
A_mat = assemble_matrix(a, bcs=[bc])
A_mat.assemble()


b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])

uh = fem.Function(H_space)

num_cells = domain.topology.index_map(domain.topology.dim).size_local

ams_opts = {
    "ksp_atol": 1e-6,
    "ksp_rtol": 1e-8,
    "ksp_type": "cg",
    "ksp_max_it": 50,
    "ksp_monitor_true_residual": None,
    "ksp_norm_type": "unpreconditioned",
    "pc_hypre_ams_cycle_type": 7,
    "pc_hypre_ams_tol": 0.0,  # Default is 1e-6 but we set it to 0.0 for AMS to be used as preconditioner
    "pc_hypre_ams_max_iter": 1,  # Set to 1 to use AMS as a preconditioner
    "pc_hypre_ams_print_level": 1,
    "pc_hypre_ams_amg_alpha_options": "10,1,6,6,4",
    "pc_hypre_ams_amg_beta_options": "10,1,6,6,4",
    "pc_hypre_ams_relax_type": 2,
    "pc_hypre_ams_relax_weight": 1.0,
    "pc_hypre_ams_relax_times": 1,
    "pc_hypre_ams_omega": 1.0,
}

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat)
ksp.setOptionsPrefix(f"ksp_{id(ksp)}")

opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)
for option, value in ams_opts.items():
    opts[option] = value
opts.prefixPop()

pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("ams")

G = discrete_gradient(V_CG._cpp_object, H_space._cpp_object)
G.assemble()
pc.setHYPREDiscreteGradient(G)

if degree == 1:
    cvec_0 = Function(H_space)
    cvec_0.interpolate(
        lambda x: np.vstack(
            (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_1 = Function(H_space)
    cvec_1.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_2 = Function(H_space)
    cvec_2.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]))
        )
    )

    pc.setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )
else:
    Vec_CG = fem.functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
    Pi = interpolation_matrix(Vec_CG._cpp_object, H_space._cpp_object)
    Pi.assemble()

    # Attach discrete gradient to preconditioner
    pc.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)


ksp.setFromOptions()
ksp.setUp()
pc.setUp()

par_print(comm, f"Norm of h_n: {L2_norm(h_n)}")
par_print(comm, f"Norm of B: {L2_norm(mu * h_n)}")

vector_vis = fem.functionspace(
    domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
)

H_vis = Function(vector_vis)
H_file = VTXWriter(domain.comm, "H_field_hform.bp", H_vis, "BP4")
H_vis.interpolate(h_n)
H_file.write(t)

B = mu * h_n
B_vis = Function(vector_vis)
B_file = VTXWriter(domain.comm, "B_field_hform.bp", B_vis, "BP4")
Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
B_vis.interpolate(Bexpr)
B_file.write(t)

J_ind = curl(h_n)
J_vis = Function(vector_vis)
Jexpr = fem.Expression(J_ind, vector_vis.element.interpolation_points)
J_vis.interpolate(Jexpr)
J_file = VTXWriter(domain.comm, "J_field_hform.bp", J_vis, "BP4")
J_file.write(t)

E = rho * J_ind
E_vis = Function(vector_vis)
Eexpr = fem.Expression(E, vector_vis.element.interpolation_points)
E_vis.interpolate(Eexpr)
E_file = VTXWriter(domain.comm, "E_field_hform.bp", E_vis, "BP4")
E_file.write(t)


output_freq = 1  # Frequency of writing output

for n in range(num_steps):

    par_print(comm, f"t is {t.expression().value}")

    t.expression().value += d_t
    H_bc.interpolate(H_applied(t.expression().value))

    # One linear solve per time step: no Picard/Newton iteration needed.
    with b.localForm() as loc_b:
        loc_b.set(0)
    petsc.assemble_vector(b, L)

    petsc.apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    par_print(comm, f"KSP its {ksp.getIterationNumber()}, reason {ksp.getConvergedReason()}")

    h_n.x.array[:] = uh.x.array
    h_n.x.scatter_forward()


    par_print(comm, f"Norm of h_n: {L2_norm(h_n)}")
    par_print(comm, f"Norm of B: {L2_norm(B)}")
    par_print(comm, f"Norm of E: {np.sqrt(comm.allreduce(fem.assemble_scalar(form(inner(E, E) * dx)), op=MPI.SUM))}")
    par_print(comm, f"Norm of J: {L2_norm(J_ind)}")

    if (n + 1) % output_freq == 0:
        par_print(comm, "Writing output ")
        H_vis.interpolate(h_n)
        H_file.write(t)

        B_vis.interpolate(Bexpr)
        B_file.write(t)

        J_vis.interpolate(Jexpr)
        J_file.write(t)

        E_vis.interpolate(Eexpr)
        E_file.write(t)

# %%
