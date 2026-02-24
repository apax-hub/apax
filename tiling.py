import jax
import jax.numpy as jnp

# Constants
TILE_SIZE = 32


def compute_inverse_3x3(h):
    """
    Computes the analytic inverse of a 3x3 matrix h.
    Robust replacement for jnp.linalg.inv(h) to avoid cuSolver errors.
    """
    # Unpack for readability (and compiler optimization)
    a, b, c = h[0,0], h[0,1], h[0,2]
    d, e, f = h[1,0], h[1,1], h[1,2]
    g, h_, i = h[2,0], h[2,1], h[2,2] # h_ to avoid shadowing function arg

    # Determinant (Triple product)
    det = a * (e * i - f * h_) - b * (d * i - f * g) + c * (d * h_ - e * g)
    inv_det = 1.0 / det

    # Cofactor expansion (Adjugate matrix)
    # Row 0
    res_00 = (e * i - f * h_) * inv_det
    res_01 = (c * h_ - b * i) * inv_det
    res_02 = (b * f - c * e) * inv_det
    
    # Row 1
    res_10 = (f * g - d * i) * inv_det
    res_11 = (a * i - c * g) * inv_det
    res_12 = (c * d - a * f) * inv_det
    
    # Row 2
    res_20 = (d * h_ - e * g) * inv_det
    res_21 = (b * g - a * h_) * inv_det
    res_22 = (a * e - b * d) * inv_det

    return jnp.array([
        [res_00, res_01, res_02],
        [res_10, res_11, res_12],
        [res_20, res_21, res_22]
    ])

def to_fractional(R, box):
    # Case 1: Cubic/Orthogonal (1D box)
    if box.ndim == 1:
        return R / box
    
    # Case 2: Triclinic (3x3 box)
    inv_box = compute_inverse_3x3(box)
    
    # R = S @ box  =>  S = R @ inv_box
    # We do NOT transpose inv_box here.
    # Since 'box' rows are the lattice vectors, 'inv_box' columns are the reciprocal vectors.
    # Multiplying R (N,3) by inv_box (3,3) correctly projects R onto the reciprocal axes.
    return jnp.dot(R, inv_box)


def sort_and_tile(R, Z, box):
    """
    R: (N, 3)
    Z: (N,)
    box: (3, 3) or (3,)
    """
    N = R.shape[0]
    
    # 1. Sort using Triclinic-aware Morton Codes
    keys = compute_morton_code(R, box)
    perm = jnp.argsort(keys)
    R_sorted = R[perm]
    Z_sorted = Z[perm]
    
    # 2. Pad to multiple of TILE_SIZE
    remainder = N % TILE_SIZE
    if remainder != 0:
        n_pad = TILE_SIZE - remainder
        # Pad coordinates to infinity
        R_pad = jnp.ones((n_pad, 3)) * 1e6
        # Pad species to 0
        Z_pad = jnp.zeros((n_pad,), dtype=Z.dtype)
        
        R_sorted = jnp.concatenate([R_sorted, R_pad], axis=0)
        Z_sorted = jnp.concatenate([Z_sorted, Z_pad], axis=0)
        
    # 3. Reshape
    n_tiles = R_sorted.shape[0] // TILE_SIZE
    R_tiled = R_sorted.reshape(n_tiles, TILE_SIZE, 3)
    Z_tiled = Z_sorted.reshape(n_tiles, TILE_SIZE)
    
    return R_tiled, Z_tiled, perm


import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ase.build import bulk

# Import the updated logic
# (Assuming you saved the code above as `tiling.py`)
# from tiling import sort_and_tile
# For this script to be standalone, I will paste the core sort logic here:


def compute_morton_code(R, box):
    S = to_fractional(R, box)
    S = S - jnp.floor(S)
    grid_idx = (S * 1024).astype(jnp.uint32)
    def spread(x):
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x <<  8)) & 0x0300F00F
        x = (x | (x <<  4)) & 0x030C30C3
        x = (x | (x <<  2)) & 0x09249249
        return x
    return spread(grid_idx[:, 0]) | (spread(grid_idx[:, 1]) << 1) | (spread(grid_idx[:, 2]) << 2)

# --- 1. Setup Skewed System ---



# Create Zinc (Hexagonal Close Packed) - This has a naturally skewed cell
atoms = bulk('Zn', 'hcp', a=2.66, c=4.95) * (4, 4, 3)
atoms.rattle(0.05)

R_orig = jnp.array(atoms.get_positions())
box = jnp.array(atoms.cell.array) # This will be 3x3 with off-diagonal terms

print(f"Box Shape: {box.shape}")
print(f"Box Matrix:\n{box}")

# --- 2. Run Sorting ---

keys = compute_morton_code(R_orig, box)
perm = jnp.argsort(keys)
R_sorted = R_orig[perm]

# --- 3. Visualization Helpers ---

def plot_box(ax, box, color='k'):
    """Draws the 12 edges of a generic 3x3 box."""
    # The 8 corners of the parallelepiped in Fractional coords
    corners_frac = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1],
        [1,1,0], [1,0,1], [0,1,1], [1,1,1]
    ])
    # Transform to Cartesian
    corners = np.dot(corners_frac, box)
    
    # Define edges by connecting corner indices
    edges = [
        [0,1], [0,2], [0,3], # Origin to axes
        [1,4], [2,4],        # xy face
        [1,5], [3,5],        # xz face
        [2,6], [3,6],        # yz face
        [4,7], [5,7], [6,7]  # Far corner
    ]
    
    for start, end in edges:
        p1 = corners[start]
        p2 = corners[end]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=0.3)

# --- 4. Plot ---

fig = plt.figure(figsize=(14, 6))

# Plot 1: Unsorted
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_box(ax1, box)
ax1.scatter(R_orig[:,0], R_orig[:,1], R_orig[:,2], c=np.arange(len(R_orig)), cmap='viridis', s=15, alpha=0.8)
ax1.plot(R_orig[:,0], R_orig[:,1], R_orig[:,2], color='gray', alpha=0.2, linewidth=0.5)
ax1.set_title("Original Order (ASE Default)")
ax1.set_xlabel('x')

# Plot 2: Morton Sorted
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_box(ax2, box)
# The line connects i -> i+1 in sorted order
ax2.plot(R_sorted[:,0], R_sorted[:,1], R_sorted[:,2], color='black', alpha=0.6, linewidth=1.0)
p2 = ax2.scatter(R_sorted[:,0], R_sorted[:,1], R_sorted[:,2], c=np.arange(len(R_sorted)), cmap='viridis', s=15, alpha=1.0)
ax2.set_title("Fractional Morton Order\n(Should follow skewed lattice)")

plt.colorbar(p2, ax=ax2, label='Memory Index')
plt.tight_layout()
plt.show()
# plt.savefig("triclinic_morton_sort.png", dpi=300)