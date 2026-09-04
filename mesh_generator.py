import numpy as np
import os
from PIL import Image


# ============================================================
# DEPTH WIZARD
# HIGH DETAIL + SOLID 3D TERRAIN MESH GENERATOR
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

RDSM_FILE = "output/rDSM.npy"
IMAGE_FILE = "output/original.png"

OBJ_FILE = "output/terrain_mesh.obj"
MTL_FILE = "output/terrain_mesh.mtl"


# ============================================================
# MAIN SETTINGS
# ============================================================

# ------------------------------------------------------------
# ELEVATION DIRECTION
# ------------------------------------------------------------

# IMPORTANT:
# Keep this FALSE.
#
# The X/Z spatial orientation is already correct and this
# setting controls whether elevation itself is inverted.
#
INVERT_ELEVATION = False


# ------------------------------------------------------------
# VERTICAL EXAGGERATION
# ------------------------------------------------------------

# Larger value = taller terrain.
#
# 10  = subtle
# 20  = noticeable
# 35  = strong
# 50  = very strong
#
HEIGHT_SCALE = 35.0


# ------------------------------------------------------------
# DETAIL / SUBDIVISION
# ------------------------------------------------------------

# 1 = original rDSM resolution
# 2 = 2x resolution
# 3 = 3x resolution
#
# 2 is recommended.
#
# This uses bilinear interpolation between real rDSM samples.
#
SUBDIVISION = 2


# ------------------------------------------------------------
# SMOOTHING
# ------------------------------------------------------------

# Keep FALSE to preserve the original elevation detail.
#
ENABLE_SMOOTHING = False

SMOOTHING_ITERATIONS = 1


# ------------------------------------------------------------
# BASE
# ------------------------------------------------------------

# Thickness of the solid base.
#
# The terrain surface starts at Y >= 0.
# The base extends downward from Y = 0.
#
BASE_THICKNESS = 0.30


# ------------------------------------------------------------
# CENTERING
# ------------------------------------------------------------

CENTER_X = True
CENTER_Z = True


# ============================================================
# CHECK INPUT FILE
# ============================================================

print()
print("============================================================")
print(" DEPTH WIZARD")
print(" HIGH DETAIL SOLID TERRAIN GENERATOR")
print("============================================================")
print()


if not os.path.exists(RDSM_FILE):

    print("ERROR: rDSM file not found!")
    print()
    print("Expected:")
    print(RDSM_FILE)
    print()

    raise SystemExit


# ============================================================
# LOAD rDSM
# ============================================================

print("Loading rDSM...")

heightmap = np.load(RDSM_FILE)

print("rDSM loaded successfully.")
print()


# ============================================================
# CHECK DATA
# ============================================================

print("Input information")
print("-----------------")

print("Shape     :", heightmap.shape)
print("Data type :", heightmap.dtype)


if heightmap.ndim != 2:

    print()
    print("ERROR:")
    print("The rDSM must be a 2D array.")

    raise SystemExit


original_height, original_width = heightmap.shape


print("Width     :", original_width)
print("Height    :", original_height)


# ============================================================
# CONVERT TO FLOAT32
# ============================================================

heightmap = heightmap.astype(
    np.float32
)


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

invalid_mask = ~np.isfinite(
    heightmap
)

invalid_count = int(
    np.sum(invalid_mask)
)


if invalid_count > 0:

    print()
    print("Invalid values detected:", invalid_count)

    valid_values = heightmap[
        ~invalid_mask
    ]

    if len(valid_values) == 0:

        print(
            "ERROR: rDSM contains no valid values."
        )

        raise SystemExit


    replacement = float(
        np.median(valid_values)
    )


    heightmap[
        invalid_mask
    ] = replacement


    print(
        "Invalid values replaced with:",
        replacement
    )


# ============================================================
# ORIGINAL STATISTICS
# ============================================================

raw_min = float(
    np.min(heightmap)
)

raw_max = float(
    np.max(heightmap)
)

raw_mean = float(
    np.mean(heightmap)
)

raw_std = float(
    np.std(heightmap)
)

raw_range = raw_max - raw_min


print()
print("rDSM statistics")
print("----------------")

print(
    "Minimum :", raw_min
)

print(
    "Maximum :", raw_max
)

print(
    "Mean    :", raw_mean
)

print(
    "Std Dev :", raw_std
)

print(
    "Range   :", raw_range
)


if raw_range <= 0:

    print()
    print(
        "ERROR: The rDSM has no elevation variation."
    )

    raise SystemExit


# ============================================================
# OPTIONAL SMOOTHING
# ============================================================

def smooth_heightmap(
    data,
    iterations=1
):

    result = data.copy()

    for _ in range(iterations):

        padded = np.pad(
            result,
            1,
            mode="edge"
        )

        result = (

            padded[:-2, :-2] +

            padded[:-2, 1:-1] +

            padded[:-2, 2:] +

            padded[1:-1, :-2] +

            padded[1:-1, 1:-1] +

            padded[1:-1, 2:] +

            padded[2:, :-2] +

            padded[2:, 1:-1] +

            padded[2:, 2:]

        ) / 9.0


    return result


if ENABLE_SMOOTHING:

    print()
    print("Applying smoothing...")

    heightmap = smooth_heightmap(
        heightmap,
        SMOOTHING_ITERATIONS
    )

    print("Smoothing complete.")


# ============================================================
# ROBUST ELEVATION NORMALIZATION
# ============================================================

#
# Instead of blindly mapping min -> 0 and max -> 1,
# use percentiles to prevent a few extreme pixels from
# compressing the majority of the terrain.
#
# This is especially useful for real-world DSM data.
#

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0


low_value = float(
    np.percentile(
        heightmap,
        LOW_PERCENTILE
    )
)


high_value = float(
    np.percentile(
        heightmap,
        HIGH_PERCENTILE
    )
)


print()
print("Elevation normalization")
print("------------------------")

print(
    "1% percentile  :",
    low_value
)

print(
    "99% percentile :",
    high_value
)


percentile_range = (
    high_value -
    low_value
)


if percentile_range <= 0:

    print(
        "WARNING: Percentile range invalid."
    )

    print(
        "Using full elevation range instead."
    )

    low_value = raw_min
    high_value = raw_max

    percentile_range = (
        high_value -
        low_value
    )


# ------------------------------------------------------------
# NORMALIZE
# ------------------------------------------------------------

normalized = (

    heightmap -
    low_value

) / percentile_range


# ------------------------------------------------------------
# CLAMP
# ------------------------------------------------------------

normalized = np.clip(
    normalized,
    0.0,
    1.0
)


# ------------------------------------------------------------
# ELEVATION INVERSION
# ------------------------------------------------------------

if INVERT_ELEVATION:

    normalized = (
        1.0 -
        normalized
    )


# ============================================================
# VERTICAL SCALE
# ============================================================

terrain_height = (
    normalized *
    HEIGHT_SCALE
)


# ============================================================
# OPTIONAL HIGH-DETAIL SUBDIVISION
# ============================================================

#
# The original rDSM is the source of truth.
#
# SUBDIVISION = 2 means:
#
#       real pixel     interpolated     real pixel
#           |               |               |
#           ●-------●-------●-------●-------●
#
# This gives smoother geometry while preserving the original
# elevation pattern.
#


def resize_bilinear(
    data,
    factor
):

    if factor == 1:

        return data


    old_h, old_w = data.shape

    new_h = (
        (old_h - 1) *
        factor
    ) + 1

    new_w = (
        (old_w - 1) *
        factor
    ) + 1


    print()
    print(
        "Subdividing terrain..."
    )

    print(
        "Original:",
        old_w,
        "x",
        old_h
    )

    print(
        "Detailed:",
        new_w,
        "x",
        new_h
    )


    # --------------------------------------------------------
    # INTERPOLATE X DIRECTION
    # --------------------------------------------------------

    x_old = np.arange(
        old_w
    )

    x_new = np.linspace(
        0,
        old_w - 1,
        new_w
    )


    temp = np.empty(
        (old_h, new_w),
        dtype=np.float32
    )


    for y in range(old_h):

        temp[y] = np.interp(
            x_new,
            x_old,
            data[y]
        )


    # --------------------------------------------------------
    # INTERPOLATE Z/Y IMAGE DIRECTION
    # --------------------------------------------------------

    y_old = np.arange(
        old_h
    )

    y_new = np.linspace(
        0,
        old_h - 1,
        new_h
    )


    result = np.empty(
        (new_h, new_w),
        dtype=np.float32
    )


    for x in range(new_w):

        result[:, x] = np.interp(
            y_new,
            y_old,
            temp[:, x]
        )


    print(
        "Subdivision complete."
    )


    return result


terrain_height = resize_bilinear(
    terrain_height,
    SUBDIVISION
)


# ============================================================
# FINAL MESH SIZE
# ============================================================

mesh_height, mesh_width = (
    terrain_height.shape
)


print()
print("Final mesh resolution")
print("----------------------")

print(
    "Width  :",
    mesh_width
)

print(
    "Height :",
    mesh_height
)


# ============================================================
# TERRAIN DIMENSIONS
# ============================================================

#
# Coordinate system:
#
# X = image horizontal direction
# Y = elevation
# Z = image vertical direction
#
# Therefore:
#
#       Y
#       ↑
#       |
#       |
#       +--------→ X
#      /
#     /
#    Z
#
#


TERRAIN_WIDTH = float(
    mesh_width - 1
)


TERRAIN_DEPTH = float(
    mesh_height - 1
)


# ============================================================
# BASE
# ============================================================

base_y = (
    -HEIGHT_SCALE *
    BASE_THICKNESS
)


print()
print("Terrain dimensions")
print("-------------------")

print(
    "X width :",
    TERRAIN_WIDTH
)

print(
    "Y height:",
    HEIGHT_SCALE
)

print(
    "Z depth :",
    TERRAIN_DEPTH
)

print(
    "Base Y  :",
    base_y
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_directory = os.path.dirname(
    OBJ_FILE
)


os.makedirs(
    output_directory,
    exist_ok=True
)


# ============================================================
# VERTICES
# ============================================================

vertices = []

uvs = []

faces = []


# ============================================================
# SURFACE INDEX MAP
# ============================================================

surface_index = np.zeros(
    (
        mesh_height,
        mesh_width
    ),
    dtype=np.int64
)


# ============================================================
# CREATE SURFACE VERTICES
# ============================================================

print()
print("Generating high-detail surface...")


for z in range(mesh_height):

    for x in range(mesh_width):

        # ----------------------------------------------------
        # CENTER X
        # ----------------------------------------------------

        if CENTER_X:

            world_x = (
                x -
                (mesh_width - 1) / 2.0
            )

        else:

            world_x = float(x)


        # ----------------------------------------------------
        # CENTER Z
        # ----------------------------------------------------

        if CENTER_Z:

            world_z = (
                z -
                (mesh_height - 1) / 2.0
            )

        else:

            world_z = float(z)


        # ----------------------------------------------------
        # HEIGHT
        # ----------------------------------------------------

        world_y = float(
            terrain_height[z, x]
        )


        # ----------------------------------------------------
        # STORE VERTEX
        # ----------------------------------------------------

        vertex_id = len(
            vertices
        )


        vertices.append(
            (
                world_x,
                world_y,
                world_z
            )
        )


        # ----------------------------------------------------
        # UV
        # ----------------------------------------------------

        if mesh_width > 1:

            u = (
                x /
                (mesh_width - 1)
            )

        else:

            u = 0.0


        if mesh_height > 1:

            v = (
                z /
                (mesh_height - 1)
            )

        else:

            v = 0.0


        #
        # Texture V is flipped here so that the image and
        # terrain maintain their previously corrected
        # correspondence.
        #

        v = 1.0 - v


        uvs.append(
            (
                u,
                v
            )
        )


        surface_index[z, x] = (
            vertex_id
        )


print(
    "Surface vertices:",
    len(vertices)
)


# ============================================================
# CREATE TERRAIN TRIANGLES
# ============================================================

print()
print("Generating terrain triangles...")


for z in range(
    mesh_height - 1
):

    for x in range(
        mesh_width - 1
    ):

        a = int(
            surface_index[z, x]
        )

        b = int(
            surface_index[z, x + 1]
        )

        c = int(
            surface_index[z + 1, x]
        )

        d = int(
            surface_index[z + 1, x + 1]
        )


        # ----------------------------------------------------
        # TRIANGLE 1
        # ----------------------------------------------------

        faces.append(
            (
                a,
                c,
                b
            )
        )


        # ----------------------------------------------------
        # TRIANGLE 2
        # ----------------------------------------------------

        faces.append(
            (
                b,
                c,
                d
            )
        )


print(
    "Terrain triangles:",
    len(faces)
)


# ============================================================
# CREATE BOTTOM VERTICES
# ============================================================

print()
print("Creating solid bottom...")


bottom_index = np.zeros(
    (
        mesh_height,
        mesh_width
    ),
    dtype=np.int64
)


for z in range(mesh_height):

    for x in range(mesh_width):

        if CENTER_X:

            world_x = (
                x -
                (mesh_width - 1) / 2.0
            )

        else:

            world_x = float(x)


        if CENTER_Z:

            world_z = (
                z -
                (mesh_height - 1) / 2.0
            )

        else:

            world_z = float(z)


        vertex_id = len(
            vertices
        )


        vertices.append(
            (
                world_x,
                base_y,
                world_z
            )
        )


        #
        # Bottom UV coordinates.
        #

        if mesh_width > 1:

            u = (
                x /
                (mesh_width - 1)
            )

        else:

            u = 0.0


        if mesh_height > 1:

            v = (
                z /
                (mesh_height - 1)
            )

        else:

            v = 0.0


        uvs.append(
            (
                u,
                1.0 - v
            )
        )


        bottom_index[z, x] = (
            vertex_id
        )


print(
    "Bottom vertices:",
    mesh_width *
    mesh_height
)


# ============================================================
# SIDE WALL FUNCTION
# ============================================================

def add_wall(
    top_a,
    top_b,
    bottom_a,
    bottom_b,
    reverse=False
):

    if not reverse:

        faces.append(
            (
                top_a,
                bottom_a,
                top_b
            )
        )

        faces.append(
            (
                top_b,
                bottom_a,
                bottom_b
            )
        )

    else:

        faces.append(
            (
                top_b,
                bottom_a,
                top_a
            )
        )

        faces.append(
            (
                top_b,
                bottom_b,
                bottom_a
            )
        )


# ============================================================
# FRONT WALL
# ============================================================

print()
print("Creating front wall...")


z = 0


for x in range(
    mesh_width - 1
):

    top_a = int(
        surface_index[z, x]
    )

    top_b = int(
        surface_index[z, x + 1]
    )

    bottom_a = int(
        bottom_index[z, x]
    )

    bottom_b = int(
        bottom_index[z, x + 1]
    )


    add_wall(
        top_a,
        top_b,
        bottom_a,
        bottom_b
    )


# ============================================================
# BACK WALL
# ============================================================

print(
    "Creating back wall..."
)


z = mesh_height - 1


for x in range(
    mesh_width - 1
):

    top_a = int(
        surface_index[z, x]
    )

    top_b = int(
        surface_index[z, x + 1]
    )

    bottom_a = int(
        bottom_index[z, x]
    )

    bottom_b = int(
        bottom_index[z, x + 1]
    )


    add_wall(
        top_a,
        top_b,
        bottom_a,
        bottom_b,
        True
    )


# ============================================================
# LEFT WALL
# ============================================================

print(
    "Creating left wall..."
)


x = 0


for z in range(
    mesh_height - 1
):

    top_a = int(
        surface_index[z, x]
    )

    top_b = int(
        surface_index[z + 1, x]
    )

    bottom_a = int(
        bottom_index[z, x]
    )

    bottom_b = int(
        bottom_index[z + 1, x]
    )


    add_wall(
        top_a,
        top_b,
        bottom_a,
        bottom_b,
        True
    )


# ============================================================
# RIGHT WALL
# ============================================================

print(
    "Creating right wall..."
)


x = mesh_width - 1


for z in range(
    mesh_height - 1
):

    top_a = int(
        surface_index[z, x]
    )

    top_b = int(
        surface_index[z + 1, x]
    )

    bottom_a = int(
        bottom_index[z, x]
    )

    bottom_b = int(
        bottom_index[z + 1, x]
    )


    add_wall(
        top_a,
        top_b,
        bottom_a,
        bottom_b
    )


print(
    "Side walls complete."
)


# ============================================================
# BOTTOM SURFACE
# ============================================================

print()
print("Creating bottom surface...")


for z in range(
    mesh_height - 1
):

    for x in range(
        mesh_width - 1
    ):

        a = int(
            bottom_index[z, x]
        )

        b = int(
            bottom_index[z, x + 1]
        )

        c = int(
            bottom_index[z + 1, x]
        )

        d = int(
            bottom_index[z + 1, x + 1]
        )


        #
        # Reverse winding because the bottom faces downward.
        #

        faces.append(
            (
                a,
                b,
                c
            )
        )


        faces.append(
            (
                b,
                d,
                c
            )
        )


print(
    "Bottom surface complete."
)


# ============================================================
# MATERIAL FILE
# ============================================================

print()
print("Creating MTL file...")


with open(
    MTL_FILE,
    "w",
    encoding="utf-8"
) as mtl:

    mtl.write(
        "# Depth Wizard Terrain Material\n"
    )

    mtl.write(
        "newmtl TerrainMaterial\n"
    )

    mtl.write(
        "Ka 1.000 1.000 1.000\n"
    )

    mtl.write(
        "Kd 1.000 1.000 1.000\n"
    )

    mtl.write(
        "Ks 0.000 0.000 0.000\n"
    )

    mtl.write(
        "Ns 10.000\n"
    )

    mtl.write(
        "d 1.000\n"
    )

    mtl.write(
        "illum 2\n"
    )

    mtl.write(
        "map_Kd original.png\n"
    )


# ============================================================
# OBJ FILE
# ============================================================

print()
print("Writing OBJ file...")


with open(
    OBJ_FILE,
    "w",
    encoding="utf-8"
) as obj:

    obj.write(
        "# ====================================================\n"
    )

    obj.write(
        "# DEPTH WIZARD HIGH DETAIL SOLID TERRAIN\n"
    )

    obj.write(
        "# ====================================================\n"
    )

    obj.write(
        "\n"
    )


    # --------------------------------------------------------
    # MATERIAL
    # --------------------------------------------------------

    obj.write(
        "mtllib terrain_mesh.mtl\n"
    )

    obj.write(
        "o DepthWizardTerrain\n"
    )

    obj.write(
        "usemtl TerrainMaterial\n"
    )

    obj.write(
        "\n"
    )


    # --------------------------------------------------------
    # VERTICES
    # --------------------------------------------------------

    print(
        "Writing vertices..."
    )


    for x, y, z in vertices:

        obj.write(
            f"v {x:.6f} {y:.6f} {z:.6f}\n"
        )


    obj.write(
        "\n"
    )


    # --------------------------------------------------------
    # UV COORDINATES
    # --------------------------------------------------------

    print(
        "Writing UV coordinates..."
    )


    for u, v in uvs:

        obj.write(
            f"vt {u:.6f} {v:.6f}\n"
        )


    obj.write(
        "\n"
    )


    # --------------------------------------------------------
    # FACES
    # --------------------------------------------------------

    print(
        "Writing faces..."
    )


    for a, b, c in faces:

        a += 1
        b += 1
        c += 1


        obj.write(
            f"f {a}/{a} {b}/{b} {c}/{c}\n"
        )


# ============================================================
# FILE SIZE
# ============================================================

obj_size = (
    os.path.getsize(
        OBJ_FILE
    )
    /
    (1024 * 1024)
)


mtl_size = (
    os.path.getsize(
        MTL_FILE
    )
    /
    1024
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("============================================================")
print(" TERRAIN GENERATION COMPLETE")
print("============================================================")
print()

print("SOURCE")
print("----------------------------")

print(
    "rDSM:",
    RDSM_FILE
)

print()


print("RESOLUTION")
print("----------------------------")

print(
    "Original:",
    original_width,
    "x",
    original_height
)

print(
    "Final:",
    mesh_width,
    "x",
    mesh_height
)

print(
    "Subdivision:",
    SUBDIVISION,
    "x"
)

print()


print("ELEVATION")
print("----------------------------")

print(
    "Raw minimum:",
    raw_min
)

print(
    "Raw maximum:",
    raw_max
)

print(
    "Raw range:",
    raw_range
)

print(
    "Height scale:",
    HEIGHT_SCALE
)

print(
    "Inverted:",
    INVERT_ELEVATION
)

print()


print("SOLID MESH")
print("----------------------------")

print(
    "Base thickness:",
    BASE_THICKNESS
)

print(
    "Base Y:",
    base_y
)

print(
    "Vertices:",
    len(vertices)
)

print(
    "Triangles:",
    len(faces)
)

print()


print("OUTPUT")
print("----------------------------")

print(
    "OBJ:",
    OBJ_FILE
)

print(
    "MTL:",
    MTL_FILE
)

print(
    "OBJ size:",
    f"{obj_size:.2f} MB"
)

print(
    "MTL size:",
    f"{mtl_size:.2f} KB"
)

print()


print("============================================================")
print(" DEPTH WIZARD READY")
print("============================================================")
print()