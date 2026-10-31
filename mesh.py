#%%
import gmsh
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

comm = MPI.COMM_WORLD
start = 0.005
div = 16  # Number of divisions -> h = 0.3125 mm, 16 cells across R
h = start / div

# --- Outer Box Parameters ---
Lx = 0.2    # length in x-direction (half-width 0.1 m = 20 x rod radius)
Ly = 0.2    # length in y-direction
Lz = 0.1    # height in z-direction (COMSOL z range -0.05 .. 0.05)
lc = 1.0

boundary_tags = {
    "cube_boundary": 1,
    "upper_surface": 2,
    "side_surface": 3,
    "bottom_surface": 4
}

# --- Inner Cylinder Parameters ---
cyl_center_x, cyl_center_y = Lx / 2, Ly / 2
cyl_radius = 0.005                     # 5 mm
cyl_height = 0.01                      # 10 mm, a 1:1 cylinder (H/D = 1.0)
cyl_base_z = (Lz - cyl_height) / 2.0   # centred in z, so both caps are interior

gmsh.initialize()

# --- Outer Box ---
p1 = gmsh.model.occ.addPoint(0, 0, 0, lc)
p2 = gmsh.model.occ.addPoint(Lx, 0, 0, lc)
p3 = gmsh.model.occ.addPoint(Lx, Ly, 0, lc)
p4 = gmsh.model.occ.addPoint(0, Ly, 0, lc)

cl = gmsh.model.occ.addCurveLoop([
    gmsh.model.occ.addLine(p1, p2),
    gmsh.model.occ.addLine(p2, p3),
    gmsh.model.occ.addLine(p3, p4),
    gmsh.model.occ.addLine(p4, p1),
])

surface = gmsh.model.occ.addPlaneSurface([cl])
extrude_result = gmsh.model.occ.extrude([(2, surface)], 0, 0, Lz)
outer_volume = extrude_result[1]  # (3, vol_tag)

# --- Inner Cylinder ---
circle = gmsh.model.occ.addCircle(cyl_center_x, cyl_center_y, cyl_base_z, cyl_radius)
curve_loop = gmsh.model.occ.addCurveLoop([circle])
cyl_base_surface = gmsh.model.occ.addPlaneSurface([curve_loop])
extrude_result_inner = gmsh.model.occ.extrude([(2, cyl_base_surface)], 0, 0, cyl_height)
inner_volume = extrude_result_inner[1]  # (3, vol_tag)

# --- Boolean Fragment (Split box and cylinder) ---
model_dim_tags = gmsh.model.occ.fragment(
    [(3, outer_volume[1])],
    [(3, inner_volume[1])]
)
gmsh.model.occ.synchronize()

# --- Identify volumes by size (fragment output order differs between gmsh versions) ---
vols = sorted(model_dim_tags[0], key=lambda v: gmsh.model.occ.getMass(*v))
cyl_vol, box_vol = vols[0], vols[1]

# --- Tagging volumes (Physical groups for subdomains) ---
gmsh.model.addPhysicalGroup(3, [cyl_vol[1]], tag=1)  # Cylinder
gmsh.model.addPhysicalGroup(3, [box_vol[1]], tag=2)  # Remaining box

# --- Boundary surfaces ---
boundary_ids_cube = [b[1] for b in gmsh.model.getBoundary([box_vol], oriented=False)]
boundary_ids_cylinder = [b[1] for b in gmsh.model.getBoundary([cyl_vol], oriented=False)]

# Classify by z-extent: the caps are flat in z, the side is not.
tol = 1e-6
top_ids, bottom_ids, side_ids = [], [], []
for s in boundary_ids_cylinder:
    zmin, zmax = gmsh.model.getBoundingBox(2, s)[2::3]
    if abs(zmax - zmin) > tol:
        side_ids.append(s)
    elif abs(zmin - (cyl_base_z + cyl_height)) < tol:
        top_ids.append(s)
    elif abs(zmin - cyl_base_z) < tol:
        bottom_ids.append(s)
outer_box_ids = [s for s in boundary_ids_cube if s not in boundary_ids_cylinder]

cyl_surface_tags = boundary_ids_cylinder

# --- Mesh refinement near the cylinder ---
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", cyl_surface_tags)

r = cyl_radius
LcMin = h              # fine near the rod: this is what `div` controls
LcMax = 1.0 * r        # 5 mm
DistMin = 0.5 * r
DistMax = 4.0 * r

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", LcMin)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", LcMax)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", DistMin)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", DistMax)

# Uniform LcMin across the whole rod volume, independent of the distance ramp.
rod_field = gmsh.model.mesh.field.add("Constant")
gmsh.model.mesh.field.setNumbers(rod_field, "VolumesList", [cyl_vol[1]])
gmsh.model.mesh.field.setNumber(rod_field, "VIn", LcMin)
gmsh.model.mesh.field.setNumber(rod_field, "VOut", 1e22)

minf = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minf, "FieldsList", [threshold, rod_field])
gmsh.model.mesh.field.setAsBackgroundMesh(minf)

gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

# --- Boundary Physical Groups ---
gmsh.model.addPhysicalGroup(2, outer_box_ids, tag=boundary_tags["cube_boundary"])  # Box outer faces
gmsh.model.addPhysicalGroup(2, side_ids, tag=boundary_tags["side_surface"])         # Cylinder side
gmsh.model.addPhysicalGroup(2, top_ids, tag=boundary_tags["upper_surface"])         # Cylinder top
gmsh.model.addPhysicalGroup(2, bottom_ids, tag=boundary_tags["bottom_surface"])     # Cylinder bottom

# --- Mesh generation ---
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
gmsh.model.mesh.generate(3)

model_rank = 0
mesh_comm = comm
mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank)
mesh = mesh_data[0]
ct = mesh_data[1]
ft = mesh_data[2]
ct.name = "ct"
ft.name = "ft"

with XDMFFile(mesh.comm, "rod.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)
    xdmf.write_meshtags(ft, mesh.geometry)

gmsh.finalize()

print(f"Number of elements in ct with tag 1: {len(ct.find(1))}")
print(f"Number of elements in ct with tag 2: {len(ct.find(2))}")

print(f"Number of elements in ft with tag 1: {len(ft.find(1))}")
print(f"Number of elements in ft with tag 2: {len(ft.find(2))}")
print(f"Number of elements in ft with tag 3: {len(ft.find(3))}")
print(f"Number of elements in ft with tag 4: {len(ft.find(4))}")
