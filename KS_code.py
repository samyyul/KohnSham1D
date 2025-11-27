# %%

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Input helpers
# =========================

def get_input(prompt, default_val=None):
    """Return user input if provided, else the default."""
    if default_val is not None:
        user_in = input(f"{prompt} [{default_val}]: ").strip()
    else:
        user_in = input(f"{prompt}: ").strip()
    return user_in if user_in else default_val


def get_user_inputs():
    """Ask for number of electrons and grid points, return them as ints."""
    n_electrons = int(get_input("Enter number of electrons", "2"))
    grid_points = int(get_input("Enter number of grid points", "200"))
    return n_electrons, grid_points


# =========================
# Grid / derivatives / operators
# =========================

def build_grid(grid_points, x_min=-5.0, x_max=5.0):
    x = np.linspace(x_min, x_max, grid_points)
    return x


def build_derivative_matrices(x):
    """
    Build first- and second-derivative finite-difference matrices D and D2,
    and the kinetic operator T_hat = -1/2 d^2/dx^2.
    """
    n = len(x)
    h = x[1] - x[0]

    # First derivative (forward difference)
    D = -np.eye(n) + np.diag(np.ones(n - 1), k=1)
    D /= h

    # Second derivative (central difference via D * (-D^T))
    D2 = D @ (-D.T)
    D2[-1, -1] = D2[0, 0]  # crude boundary fix

    T_hat = -0.5 * D2
    return D, D2, T_hat


def solve_harmonic_oscillator(x, T_hat):
    """Hamiltonian and eigenstates for v_ext(x) = x^2."""
    v_ext = x**2
    H_hat = T_hat + np.diag(v_ext)
    eig_harm, psi_harm = np.linalg.eigh(H_hat)
    return eig_harm, psi_harm


# =========================
# Basic numerical helpers
# =========================

def integral(x, y):
    """Simple 1D integral on an equally spaced grid."""
    dx = x[1] - x[0]
    return np.sum(y * dx, axis=0)


def occupations(n_electrons):
    """Hartree–Fock-style occupancies 0/1/2."""
    occs = [2 for _ in range(n_electrons // 2)]
    if n_electrons % 2:
        occs.append(1)
    return np.array(occs, dtype=float)


def density(psi, n_electrons, x):
    """
    Build electron density from KS orbitals psi(x, n)
    using occupations f_n and normalised orbitals.
    """
    # normalise orbitals
    I = integral(x, psi**2)             # shape (n_orbitals,)
    norm_psi = psi / np.sqrt(I)[None, :]  # broadcast over x

    occs = occupations(n_electrons)
    rho = np.zeros(norm_psi.shape[0])

    for occ, orb in zip(occs, np.abs(norm_psi.T)):
        rho += occ * orb**2

    # sanity check:
    # print("N electrons from rho:", integral(x, rho))
    return rho


# =========================
# XC and Hartree functionals
# =========================

def calculate_exchange(rho, x):
    """
    LDA exchange energy and potential for a 3D-like kernel
    used in a 1D toy model.
    """
    dx = x[1] - x[0]
    const = (3.0 / np.pi)**(1.0 / 3.0)

    energy = -0.75 * const * np.sum(rho**(4.0 / 3.0) * dx)
    potential = -const * rho**(1.0 / 3.0)
    return energy, potential


def calculate_coulomb(rho, x):
    """
    Hartree (Coulomb) energy and potential with softened 1D kernel 1/(|x-x'|+1).
    """
    dx = x[1] - x[0]
    diff = np.abs(x[:, None] - x[None, :]) + 1.0

    energy = np.sum(
        rho[:, None] * rho[None, :] * 0.5 * dx * dx / diff
    )
    potential = np.sum(
        rho[None, :] * dx / diff,
        axis=1
    )
    return energy, potential


# =========================
# KS total energy
# =========================

def ks_total_energy(rho, eig_ks, n_electrons, x):
    dx = x[1] - x[0]
    occs = occupations(n_electrons)

    # Sum over occupied KS eigenvalues
    eps_sum = np.sum(occs * eig_ks[:len(occs)])

    # External potential (harmonic)
    v_ext = x**2
    int_n_vext = np.sum(rho * v_ext * dx)

    # Hartree and exchange parts
    E_H, v_H = calculate_coulomb(rho, x)
    E_X, v_X = calculate_exchange(rho, x)

    # ∫ n(x) v_xc(x) dx  (here xc = X only)
    int_n_vxc = np.sum(rho * v_X * dx)

    E_tot = eps_sum - int_n_vext + E_H + E_X + int_n_vxc

    pieces = {
        "eps_sum": eps_sum,
        "int_n_vext": int_n_vext,
        "E_H": E_H,
        "E_X": E_X,
        "int_n_vxc": int_n_vxc,
    }
    return E_tot, pieces


# =========================
# SCF machinery
# =========================

def density_function(n_electrons, rho, x, T_hat):
    """Given a trial density, build KS Hamiltonian, solve, and return new density."""
    # Hartree + exchange potentials from rho
    _, v_X = calculate_exchange(rho, x)
    _, v_H = calculate_coulomb(rho, x)

    v_ext = x**2
    v_total = v_ext + v_H + v_X

    H_hat = T_hat + np.diag(v_total)
    eig_ks, psi_ks = np.linalg.eigh(H_hat)

    rho_new = density(psi_ks, n_electrons, x)
    return rho_new, eig_ks, psi_ks


def self_consistent_loop(n_electrons, x, T_hat, max_iter=100, tol=1e-6):
    # initial guess: zero density
    rho = np.zeros_like(x)

    for iteration in range(max_iter):
        rho_new, eig_ks, psi_ks = density_function(n_electrons, rho, x, T_hat)
        error = np.linalg.norm(rho_new - rho)

        print(f"Iteration {iteration}: error = {error:.6e}")
        print("  band energy (sum eps):", np.sum(eig_ks[:n_electrons]))

        rho = rho_new
        if error < tol:
            print(f"Converged after {iteration} iterations.\n")
            break

    return rho, eig_ks, psi_ks


# =========================
# Driver
# =========================

def run_ks_scf():
    n_electrons, grid_points = get_user_inputs()
    x = build_grid(grid_points)
    D, D2, T_hat = build_derivative_matrices(x)

    # Optional: test derivative operators
    # y = np.sin(x)
    # plt.plot(x, y, label="f(x)")
    # plt.plot(x[1:-1], (D @ y)[1:-1], label="f'(x)")
    # plt.plot(x[1:-1], (D2 @ y)[1:-1], label="f''(x)")
    # plt.legend()
    # plt.show()

    # Free-particle spectrum (kinetic only) if you want it:
    eig_free, psi_free = np.linalg.eigh(T_hat)

    # Harmonic oscillator without interaction
    eig_harm, psi_harm = solve_harmonic_oscillator(x, T_hat)

    # Interacting KS SCF
    rho_scf, eig_scf, psi_scf = self_consistent_loop(n_electrons, x, T_hat)
    E_tot, pieces = ks_total_energy(rho_scf, eig_scf, n_electrons, x)

    print("Total KS energy:", E_tot)
    print("  sum f_n eps_n      =", pieces["eps_sum"])
    print("  -∫ n v_ext         =", -pieces["int_n_vext"])
    print("  E_Coulomb          =", pieces["E_H"])
    print("  E_Exchange         =", pieces["E_X"])
    print("  ∫ n v_x            =", pieces["int_n_vxc"])

    # Compare densities: SCF vs non-interacting HO
    rho_harm = density(psi_harm, n_electrons, x)

    plt.plot(x, rho_scf, label="SCF density")
    plt.plot(x, rho_harm, label="harmonic (no interaction)")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("n(x)")
    plt.show()

    # Plot KS eigenvalues
    plt.plot(eig_scf, marker="o", linestyle="", label="KS eigenvalues")
    plt.legend()
    plt.xlabel("orbital index")
    plt.ylabel("ε_n")
    plt.show()


if __name__ == "__main__":
    run_ks_scf()