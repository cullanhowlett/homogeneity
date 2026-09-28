import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

# Set random seed for reproducibility
np.random.seed(42)

# Box parameters
box_size = 4000  # Mpc/h (from -1000 to 1000)
box_min = -2000
box_max = 2000
n_particles = 100_000

print("Generating particle distributions...")

# ========================================
# 1. UNIFORM DISTRIBUTION
# ========================================
print("1. Generating uniform distribution...")
uniform_particles = np.random.uniform(box_min, box_max, size=(n_particles, 3))

# ========================================
# 2. GAUSSIAN DISTRIBUTION
# ========================================
print("2. Generating Gaussian distribution...")
sigma = 400  # Mpc/h width
gaussian_particles = np.random.normal(0, sigma, size=(n_particles, 3))
mask = np.all((gaussian_particles >= box_min) & (gaussian_particles <= box_max), axis=1)
gaussian_particles = gaussian_particles[mask]
print(f"   Gaussian particles in box: {len(gaussian_particles)}")

# ========================================
# 3. VOID FOAM DISTRIBUTION (power-law spheres)
# ========================================
print("3. Generating void foam distribution...")
# Voids with radii from P(r) ~ r^{-alpha}. Particles rejected if inside
# any void -> matter concentrates on void walls, like the cosmic web.
# alpha=2.0, N=5000 gives ~70% void volume fraction.

n_voids = 5000
r_min_void = 20.0  # Mpc/h
r_max_void = 200.0  # Mpc/h
alpha = 3.0

# Void centres and radii
void_centres = np.random.uniform(box_min, box_max, size=(n_voids, 3))

# Inverse CDF for P(r) ~ r^{-alpha}
u = np.random.uniform(size=n_voids)
void_radii = (r_min_void ** (1 - alpha) + u * (r_max_void ** (1 - alpha) - r_min_void ** (1 - alpha))) ** (
    1.0 / (1 - alpha)
)

print(
    f"   Void radii: min={void_radii.min():.1f}, max={void_radii.max():.1f}, "
    f"median={np.median(void_radii):.1f} Mpc/h"
)

# Build a KDTree on void centres. For each candidate particle, query only
# voids within r_max_void — then check the actual per-void radii.
# This avoids any large N_candidates x N_voids arrays.
void_tree = KDTree(void_centres)

fractal_particles = []
batch_size = 200_000

while len(fractal_particles) < n_particles:
    n_needed = n_particles - len(fractal_particles)
    n_batch = min(int(n_needed * 1.5), batch_size)
    candidates = np.random.uniform(box_min, box_max, size=(n_batch, 3))

    # For each candidate, get indices of void centres within r_max_void
    neighbour_indices, neighbour_dists = void_tree.query_radius(candidates, r=r_max_void, return_distance=True)

    # Check each candidate: is it inside any of its nearby voids?
    outside = np.ones(n_batch, dtype=bool)
    for i in range(n_batch):
        if len(neighbour_indices[i]) > 0:
            # Compare actual distances to the specific radii of those voids
            if np.any(neighbour_dists[i] < void_radii[neighbour_indices[i]]):
                outside[i] = False

    fractal_particles.extend(candidates[outside])
    print(f"   Batch: {np.sum(outside):,} kept / {n_batch:,} candidates " f"(total so far: {len(fractal_particles):,})")

fractal_particles = np.array(fractal_particles[:n_particles])
print(f"   Void foam particles: {len(fractal_particles):,}")

# ========================================
# 4. UNIFORM WITH VOID (GAUSSIAN PROFILE)
# ========================================
print("4. Generating uniform distribution with Gaussian void profile...")
void_sigma = 800  # Mpc/h - wide void, empty centre
# p(r) = 1 - exp(-(r/sigma)^2)

n_generate = int(n_particles * 2)
void_particles = []

while len(void_particles) < n_particles:
    batch = np.random.uniform(box_min, box_max, size=(n_generate, 3))
    distances = np.sqrt(np.sum(batch**2, axis=1))
    acceptance_prob = 1.0 - np.exp(-((distances / void_sigma) ** 4))
    keep_mask = np.random.rand(len(batch)) < acceptance_prob
    void_particles.extend(batch[keep_mask])

void_particles = np.array(void_particles[:n_particles])
print(f"   Void particles: {len(void_particles)}")

# ========================================
# PLOT 1D DISTRIBUTIONS
# ========================================
print("\nPlotting 1D distributions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
tolerance = 50  # Mpc/h

distributions = [
    (uniform_particles, "Uniform Distribution"),
    (gaussian_particles, "Gaussian Distribution (σ=400 Mpc/h)"),
    (fractal_particles, "Void Foam (power-law radii, α=2.0)"),
    (void_particles, "Uniform with Gaussian Void (σ=400 Mpc/h, empty centre)"),
]

for idx, (particles, title) in enumerate(distributions):
    ax = axes[idx]
    mask = (np.abs(particles[:, 1]) < tolerance) & (np.abs(particles[:, 2]) < tolerance)
    x_slice = particles[mask, 0]
    bins = np.linspace(box_min, box_max, 100)
    counts, edges = np.histogram(x_slice, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(bin_centers, counts, linewidth=1.5)
    ax.set_xlabel("x [Mpc/h]", fontsize=11)
    ax.set_ylabel("Number of particles", fontsize=11)
    ax.set_title(f"{title}\n1D slice through y=0, z=0 (±{tolerance} Mpc/h)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(box_min, box_max)

plt.tight_layout()
plt.savefig("./1d_distributions.png", dpi=300, bbox_inches="tight")
print("Saved 1D distribution plot")

# ========================================
# SAVE PARTICLE DISTRIBUTIONS
# ========================================
print("\nSaving particle distributions...")
np.save("./uniform_particles.npy", uniform_particles)
np.save("./gaussian_particles.npy", gaussian_particles)
np.save("./fractal_particles.npy", fractal_particles)
np.save("./void_particles.npy", void_particles)

print("\nSummary:")
print(f"Uniform particles:    {len(uniform_particles):,}")
print(f"Gaussian particles:   {len(gaussian_particles):,}")
print(f"Void foam particles:  {len(fractal_particles):,}")
print(f"Void particles:       {len(void_particles):,}")
print("\nAll distributions saved as .npy files")
print("1D distributions plotted and saved")
