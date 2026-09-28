#!/usr/bin/env pytho#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSC calculation for toy models using generated uniform randoms
Much simpler and faster than edge corrections!
"""

import numpy as np
import time
from sklearn.neighbors import KDTree
from multiprocessing import Pool, current_process
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


def Nr(begin, end, tree_main, tree_rand, coords_main, weights_main, weights_rand, radii, radius_bins, rand_density):
    """
    Calculate MSC - same function as your original NERSC code
    """

    print(current_process(), begin, end)
    with np.errstate(divide="ignore"):
        inds, dists = tree_main.query_radius(coords_main[begin:end], r=radii[-1], return_distance=True)
        weighted_counts_main = np.array(
            [
                np.cumsum(np.histogram(dist, radius_bins, weights=weights_main[ind])[0])
                for (ind, dist) in zip(inds, dists)
            ]
        )
        inds, dists = tree_rand.query_radius(coords_main[begin:end], r=radii[-1], return_distance=True)
        weighted_counts_rand = np.array(
            [
                np.cumsum(np.histogram(dist, radius_bins, weights=weights_rand[ind])[0])
                for (ind, dist) in zip(inds, dists)
            ]
        )
        ratio = (weighted_counts_main * rand_density / weighted_counts_rand).T
        MSC_data = np.ma.masked_invalid(ratio * weights_main[begin:end]).sum(axis=1)
        count_data = [np.sum(weights_main[begin:end][np.isfinite(rat)]) for rat in ratio]

    return count_data, MSC_data


def radec_z_to_cartesian(ra, dec, z, origin, cosmo):
    """
    Convert RA, Dec (degrees), and redshift to Cartesian coordinates (Mpc/h).

    Parameters:
    -----------
    ra : array, shape (N,)
        Right ascension in degrees
    dec : array, shape (N,)
        Declination in degrees
    z : array, shape (N,)
        Redshift
    origin : array, shape (3,)
        Observer position in Cartesian coords
    cosmo : astropy.cosmology
        Cosmology for distance calculation

    Returns:
    --------
    coords : array, shape (N, 3)
        Cartesian coordinates [x, y, z] in Mpc/h
    """
    # Comoving distance in Mpc/h
    r_comov_Mpc = cosmo.comoving_distance(z).value  # Mpc
    r_comov = r_comov_Mpc * cosmo.h  # Mpc/h

    # Convert angles to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    # Cartesian in observer frame
    x = r_comov * np.cos(dec_rad) * np.cos(ra_rad)
    y = r_comov * np.cos(dec_rad) * np.sin(ra_rad)
    z = r_comov * np.sin(dec_rad)

    coords = np.column_stack([x, y, z])

    # Shift back to box frame
    coords += origin[np.newaxis, :]

    return coords


def cartesian_to_radec_z(coords, origin, cosmo):
    """
    Convert Cartesian coordinates (Mpc/h) to RA, Dec (degrees), and redshift.

    Parameters:
    -----------
    coords : array, shape (N, 3)
        Cartesian coordinates [x, y, z] in Mpc/h
    origin : array, shape (3,)
        Observer position in Cartesian coords
    cosmo : astropy.cosmology
        Cosmology for redshift calculation

    Returns:
    --------
    ra : array, shape (N,)
        Right ascension in degrees [0, 360)
    dec : array, shape (N,)
        Declination in degrees [-90, 90]
    z : array, shape (N,)
        Redshift
    """
    from scipy.interpolate import InterpolatedUnivariateSpline

    # Shift to observer frame
    relative = coords - origin[np.newaxis, :]

    # Comoving distance in Mpc/h
    r_comov = np.sqrt(np.sum(relative**2, axis=1))

    # RA, Dec from spherical coordinates
    x, y, z = relative[:, 0], relative[:, 1], relative[:, 2]

    ra = np.arctan2(y, x)  # radians, [-π, π]
    ra = np.degrees(ra)  # degrees
    ra = np.mod(ra, 360.0)  # [0, 360)

    dec = np.arcsin(z / (r_comov + 1e-10))  # avoid div by zero at origin
    dec = np.degrees(dec)

    # Redshift from comoving distance via inverse spline
    # Build r(z) lookup table, then invert
    z_max = np.max(r_comov) * cosmo.H0.value / 299792.458 * 1.5  # Conservative upper bound
    z_grid = np.linspace(0, z_max, 1000)
    r_grid = cosmo.comoving_distance(z_grid).value * cosmo.h  # Mpc/h

    # Create spline z(r) by swapping axes
    spline_z_of_r = InterpolatedUnivariateSpline(r_grid, z_grid, k=3)
    z = spline_z_of_r(r_comov)

    return ra, dec, z


def calculate_msc_toy_model(
    particles,
    box_size=2000,
    weights=None,
    radii=None,
    batch_size=1000,
    n_processes=8,
    n_rand_multiple=10,
    match_redshift=False,
    origin=None,
    dist_name="",
):
    """
    Calculate MSC for toy model particle distribution.

    Parameters:
    -----------
    particles : array, shape (N, 3)
        Particle positions in Cartesian coords (Mpc/h)
    box_size : float
        Size of cubic box (from -box_size/2 to +box_size/2)
    weights : array, shape (N,), optional
        Particle weights (default: all 1.0)
    radii : array, optional
        Radii at which to calculate MSC (default: 20 to 250 Mpc/h in 0.5 steps)
    batch_size : int
        Number of particles per batch for parallel processing
    n_processes : int
        Number of parallel processes
    n_rand_multiple : int
        Generate this many times more randoms than data particles
    match_redshift : bool
        If True, resample random redshifts from data distribution
    origin : array, shape (3,), optional
        Observer position for coordinate conversion (default: [0, 0, 0])

    Returns:
    --------
    radii : array
        Radii at which MSC was calculated
    MSC : array
        Mean scaled counts at each radius
    counts : array
        Total weight contributing to each radius bin
    """

    if weights is None:
        weights = np.ones(len(particles))
    if radii is None:
        radii = np.arange(20, 250, 0.5)
    if origin is None:
        origin = np.array([0.0, 0.0, 0.0])

    radius_bins = np.concatenate([[0.0], radii])

    print(f"Calculating MSC for {len(particles):,} particles")
    print(f"Box size: {box_size} Mpc/h (from {-box_size / 2} to {box_size / 2})")
    print(f"Radii: {radii[0]} to {radii[-1]} Mpc/h ({len(radii)} bins)")
    print(f"Redshift matching: {match_redshift}")
    print(f"Observer origin: {origin}")

    # Define cosmology
    cosmo = FlatLambdaCDM(H0=100.0 * u.km / u.s / u.Mpc, Om0=0.3151917236644108)

    # Calculate maximum safe radius: distance from origin to nearest box edge
    box_min = -box_size / 2
    box_max = box_size / 2
    r_max = min(
        origin[0] - box_min,  # distance to left edge
        box_max - origin[0],  # distance to right edge
        origin[1] - box_min,  # distance to front edge
        box_max - origin[1],  # distance to back edge
        origin[2] - box_min,  # distance to bottom edge
        box_max - origin[2],  # distance to top edge
    )
    print(f"Maximum radial distance for complete volume: {r_max:.2f} Mpc/h")

    # Generate randoms based on ORIGINAL data count (before cut)
    n_randoms_initial = n_rand_multiple * len(particles)
    print(f"Generating {n_randoms_initial:,} uniform randoms ({n_rand_multiple}x original data)...")
    coords_rand = np.random.uniform(box_min, box_max, size=(n_randoms_initial, 3))

    # Apply radial cut to randoms
    r_rand = np.sqrt(np.sum((coords_rand - origin[np.newaxis, :]) ** 2, axis=1))
    rand_mask = r_rand <= r_max
    coords_rand = coords_rand[rand_mask]

    print(
        f"Random particles after radial cut: {len(coords_rand):,} ({100 * len(coords_rand) / n_randoms_initial:.1f}% of original)"
    )

    # Apply radial cut to data
    r_data = np.sqrt(np.sum((particles - origin[np.newaxis, :]) ** 2, axis=1))
    data_mask = r_data <= r_max
    particles = particles[data_mask]
    weights = weights[data_mask]

    print(
        f"Data particles after radial cut: {len(particles):,} ({100 * len(particles) / len(data_mask):.1f}% of original)"
    )

    # Update n_randoms and weights to reflect actual cut random count
    n_randoms = len(coords_rand)
    weights_rand = np.ones(n_randoms)

    print(f"Effective random multiple after cuts: {n_randoms / len(particles):.2f}x data")

    if match_redshift:
        print("Converting coordinates to RA/Dec/z...")

        # Convert data to RA, Dec, z
        ra_data, dec_data, z_data = cartesian_to_radec_z(particles, origin, cosmo)

        # Convert randoms to RA, Dec, z
        ra_rand, dec_rand, z_rand = cartesian_to_radec_z(coords_rand, origin, cosmo)

        print("Resampling random redshifts from data distribution...")
        # Sample redshifts from data (with replacement)
        z_rand_matched = np.random.choice(z_data, size=len(coords_rand), replace=True)

        print("Converting back to Cartesian coordinates...")
        # Convert randoms back to Cartesian with matched redshifts
        coords_rand = radec_z_to_cartesian(ra_rand, dec_rand, z_rand_matched, origin, cosmo)

        print(f"Data redshift range: {z_data.min():.4f} - {z_data.max():.4f}")
        print(f"Random redshift range (matched): {z_rand_matched.min():.4f} - {z_rand_matched.max():.4f}")
    else:
        # Still need to compute RA/Dec/z for diagnostic plots
        ra_data, dec_data, z_data = cartesian_to_radec_z(particles, origin, cosmo)
        ra_rand, dec_rand, z_rand = cartesian_to_radec_z(coords_rand, origin, cosmo)
        z_rand_matched = z_rand  # Not actually matched, just original randoms

    # Diagnostic plots (always shown)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # RA histogram
    axes[0].hist(ra_data, bins=50, alpha=0.5, label="Data", density=True)
    axes[0].hist(ra_rand, bins=50, alpha=0.5, label="Randoms", density=True)
    axes[0].set_xlabel("RA [deg]", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("Right Ascension Distribution", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dec histogram
    axes[1].hist(dec_data, bins=50, alpha=0.5, label="Data", density=True)
    axes[1].hist(dec_rand, bins=50, alpha=0.5, label="Randoms", density=True)
    axes[1].set_xlabel("Dec [deg]", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title("Declination Distribution", fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Redshift histogram
    z_label = "Randoms (matched)" if match_redshift else "Randoms"
    axes[2].hist(z_data, bins=50, alpha=0.5, label="Data", density=True)
    axes[2].hist(z_rand_matched, bins=50, alpha=0.5, label=z_label, density=True)
    axes[2].set_xlabel("Redshift", fontsize=12)
    axes[2].set_ylabel("Density", fontsize=12)
    axes[2].set_title("Redshift Distribution", fontsize=13)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"./compute_toy_model_{dist_name}_200.png", dpi=300, bbox_inches="tight")

    print("Building KD trees...")
    tree_main = KDTree(particles)
    tree_rand = KDTree(coords_rand)

    rand_density = np.sum(weights_rand) / np.sum(weights)
    print(f"Random density: {rand_density:.4f}")

    numbatches = int(len(particles) / batch_size) + 1
    begin_array = np.arange(0, len(particles), batch_size)
    end_array = begin_array + batch_size
    end_array[-1] = len(particles)

    print(f"Processing in {numbatches} batches with {n_processes} processes...")
    start = time.time()

    with Pool(n_processes) as pool:
        processed = pool.starmap(
            Nr,
            [
                [begin, end, tree_main, tree_rand, particles, weights, weights_rand, radii, radius_bins, rand_density]
                for (begin, end) in zip(begin_array, end_array)
            ],
        )
    pool.close()
    pool.join()

    count_data = np.sum(np.vstack([process[0] for process in processed]), axis=0)
    MSC_data = np.sum(np.vstack([process[1] for process in processed]), axis=0)
    MSC_data /= count_data

    print(f"Computation time: {(time.time() - start) / 60.0:.2f} minutes")

    return radii, MSC_data, count_data


if __name__ == "__main__":

    # Process all four distributions
    distributions = [
        ("uniform", "Uniform"),
        ("gaussian", "Gaussian"),
        ("void", "Giant Void"),
        ("fractal", "Swiss Cheese"),
    ]

    for dist_name, dist_label in distributions:
        print(f"\n{'='*60}")
        print(f"Processing {dist_label}")
        print("=" * 60)

        # Load particles
        particles = np.load(f"./{dist_name}_particles.npy")

        # Calculate MSC
        radii, MSC, counts = calculate_msc_toy_model(
            particles,
            box_size=2000,
            batch_size=2000,
            n_processes=8,
            n_rand_multiple=10,
            match_redshift=False,
            origin=np.array([-200.0, -200.0, -200.0]),
            dist_name=dist_name,
        )

        # Save results
        output = np.c_[radii, MSC, counts]
        header = f"box_size=2000.0 n_particles={len(particles)} distribution={dist_name} n_rand_multiple=10"
        np.savetxt(
            f"./MSC_{dist_name}_200.txt",
            output,
            fmt="%.16f",
            delimiter=" ",
            header=header,
            comments="# ",
        )

        print(f"\nResults:")
        print(f"MSC at r=50 Mpc/h: {MSC[np.argmin(np.abs(radii-50))]:.6f}")
        print(f"MSC at r=100 Mpc/h: {MSC[np.argmin(np.abs(radii-100))]:.6f}")
        print(f"MSC at r=150 Mpc/h: {MSC[np.argmin(np.abs(radii-150))]:.6f}")
        print(f"MSC at r=200 Mpc/h: {MSC[np.argmin(np.abs(radii-200))]:.6f}")
