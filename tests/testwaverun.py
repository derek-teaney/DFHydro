"""testwaverun.py — travelling sinusoidal wave test for periodic boundary conditions.

Initialises a rightward-travelling plane wave on a periodic grid and runs it
for one transit time T = Lx/cs.  Plots the energy density on the x-midline
and y-midline at four equally-spaced times with the exact solution overlaid.

Usage:
    python testwaverun.py run  [--nx-wave N] [--ny-wave M]
    python testwaverun.py plot [--nx-wave N] [--ny-wave M]

The wave vector is  k = (nx_wave * 2pi/Lx,  ny_wave * 2pi/Ly).
Defaults: nx_wave=2, ny_wave=0  (two full waves purely in x).
"""

import vischydro as vh
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import pprint

runcommand = 'mpiexec-mpich-clang17 -n 1 ./VischydroMain.exe'

# Fixed grid / domain parameters
Lx = 1.0
Ly = 1.4
NX = 128
NY = 128
e0 = 1.0
A  = 0.02
cs = 1.0 / np.sqrt(3.0)   # cs^2 = 1/3 (conformal EOS)
T  = Lx / cs               # one x-transit time

RUN_NAME = 'sinusoidal_wave'


def wave_params(nx_wave, ny_wave):
    """Return (kx, ky, k, omega) for integer wave numbers (nx_wave, ny_wave)."""
    kx    = nx_wave * 2.0 * np.pi / Lx
    ky    = ny_wave * 2.0 * np.pi / Ly
    k     = np.sqrt(kx**2 + ky**2)
    omega = cs * k
    return kx, ky, k, omega


def ic_wave(input_data, nx_wave, ny_wave):
    """Build initial-data array for a travelling cosine wave.

    Sets charges (E, Mx, My) at t=0:
        E  = e0 + A cos(kx x + ky y)
        Mx = A cs (kx/|k|) cos(kx x + ky y)
        My = A cs (ky/|k|) cos(kx x + ky y)
    """
    kx, ky, k, _omega = wave_params(nx_wave, ny_wave)

    xarray, yarray, _dx, _dy = vh.xygrid(input_data)
    initialdata = vh.initialdata_grid(input_data)

    phase = kx * xarray + ky * yarray
    initialdata[:, :, 0] = e0 + A * np.cos(phase)              # E = Ttt
    initialdata[:, :, 1] = A * cs * (kx / k) * np.cos(phase)   # Mx = Ttx
    initialdata[:, :, 2] = A * cs * (ky / k) * np.cos(phase)   # My = Tty

    input_data["VischydroMain/initial_field_type"] = "charges"
    return initialdata


def runwave(nx_wave, ny_wave, petsc_args='', actually_run=True):
    kx, ky, k, omega = wave_params(nx_wave, ny_wave)
    print(f"Wave: nx_wave={nx_wave} ny_wave={ny_wave}  "
          f"kx={kx:.4f} ky={ky:.4f} |k|={k:.4f} omega={omega:.4f}")

    vh.input_data["Vischydro/is_periodic"]             = True
    vh.input_data["Vischydro/use_ideal_step_only"]     = True
    vh.input_data["Vischydro/is_bjorken"]              = False
    vh.input_data["Vischydro/highest_order_term_only"] = True

    vh.input_data["VischydroMain/nx"]              = NX
    vh.input_data["VischydroMain/xmin"]            = 0.0
    vh.input_data["VischydroMain/xmax"]            = Lx
    vh.input_data["VischydroMain/ny"]              = NY
    vh.input_data["VischydroMain/ymin"]            = 0.0
    vh.input_data["VischydroMain/ymax"]            = Ly

    vh.input_data["VischydroMain/eta_by_s"]        = 0.0
    vh.input_data["VischydroMain/zeta_by_s"]       = 0.0
    vh.input_data["VischydroMain/t_start"]         = 0.0
    vh.input_data["VischydroMain/t_end"]           = T
    vh.input_data["VischydroMain/print_frequency"] = 2
    vh.input_data["VischydroMain/run_name"]        = RUN_NAME

    pprint.pprint(vh.input_data.data)

    initialdata = ic_wave(vh.input_data, nx_wave, ny_wave)

    if actually_run:
        vh.runcode(initialdata, vh.input_data, runcommand=runcommand,
                   actually_run=True, petsc_args=petsc_args)
    else:
        plot_midlines(vh.input_data, RUN_NAME, nx_wave, ny_wave)
        plot_conservation(RUN_NAME)
        plt.show()


def plot_conservation(name):
    """Plot total E, Mx, My vs time, normalised to t=0 values.

    For periodic ideal hydro these are exactly conserved, so any deviation
    from 1.0 signals a boundary or numerical conservation bug.
    """
    times_arr = np.loadtxt('{}_grid_t.txt'.format(name))
    if times_arr.ndim == 1:
        times_arr = times_arr.reshape(-1, 1)
    saved_times = times_arr[:, 0]

    with h5py.File(name + '_grid.h5', 'r') as f:
        coords = f['coordinates']                      # (ny, nx, 2)
        dx = float(coords[0, 1, 0] - coords[0, 0, 0])
        dy = float(coords[1, 0, 1] - coords[0, 0, 1])
        cell_area = dx * dy

        sol = f['solution']                            # (nt, ny, nx, ndof)
        total_E  = np.sum(sol[:, :, :, 0], axis=(1, 2)) * cell_area  # Ttt
        total_Mx = np.sum(sol[:, :, :, 1], axis=(1, 2)) * cell_area  # Ttx
        total_My = np.sum(sol[:, :, :, 2], axis=(1, 2)) * cell_area  # Tty

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(saved_times, total_E  / total_E[0],  label=r'$\sum E \cdot dA \;/\; E_0$')
    ax.plot(saved_times, total_Mx / total_E[0],  label=r'$\sum M_x \cdot dA \;/\; E_0$')
    ax.plot(saved_times, total_My / total_E[0],  label=r'$\sum M_y \cdot dA \;/\; E_0$')
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('t')
    ax.set_ylabel('normalised total charge')
    ax.set_title('Conservation of E, Mx, My  (should be constant)')
    ax.legend()
    fig.tight_layout()
    outfile = name + '_conservation.pdf'
    plt.savefig(outfile)
    print(f"Saved {outfile}")


def plot_midlines(input_data, name, nx_wave, ny_wave):
    kx, ky, k, omega = wave_params(nx_wave, ny_wave)

    times_arr = np.loadtxt('{}_grid_t.txt'.format(name))
    if times_arr.ndim == 1:
        times_arr = times_arr.reshape(-1, 1)
    saved_times = times_arr[:, 0]

    # Pick saved snapshots closest to t = 0, T/3, 2T/3, T
    targets = [0.0, T / 3.0, 2.0 * T / 3.0, T]
    indices = [int(np.argmin(np.abs(saved_times - tgt))) for tgt in targets]

    ny = input_data["VischydroMain/ny"]
    nx = input_data["VischydroMain/nx"]
    jmid = ny // 2
    imid = nx // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(indices)))

    with h5py.File(name + '_grid.h5', 'r') as f:
        coords = f['coordinates']          # shape (ny, nx, 2)
        x_mid  = coords[jmid, :, 0]       # x values at y = ymid
        y_mid  = coords[:, imid, 1]       # y values at x = xmid
        ymid   = float(coords[jmid, 0, 1])
        xmid   = float(coords[0, imid, 0])

        for color, it in zip(colors, indices):
            t_snap = saved_times[it]
            sol    = f['solution'][it, :, :, 0]   # e, shape (ny, nx)
            label  = f't = {t_snap:.3f}'

            # Panel 1: e vs x at j = jmid
            ax1.plot(x_mid, sol[jmid, :], '.', color=color, label=label)
            e_exact_x = e0 + A * np.cos(kx * x_mid + ky * ymid - omega * t_snap)
            ax1.plot(x_mid, e_exact_x, '--', color=color, alpha=0.55)

            # Panel 2: e vs y at i = imid
            ax2.plot(y_mid, sol[:, imid], '.', color=color, label=label)
            e_exact_y = e0 + A * np.cos(kx * xmid + ky * y_mid - omega * t_snap)
            ax2.plot(y_mid, e_exact_y, '--', color=color, alpha=0.55)

    ax1.plot([], [], 'k--', alpha=0.55, label='exact')
    ax1.set_xlabel('x')
    ax1.set_ylabel('e')
    ax1.set_title(f'x-midline  (nx={nx_wave}, ny={ny_wave}, A={A})')
    ax1.legend()

    ax2.plot([], [], 'k--', alpha=0.55, label='exact')
    ax2.set_xlabel('y')
    ax2.set_ylabel('e')
    ax2.set_title(f'y-midline  (nx={nx_wave}, ny={ny_wave}, A={A})')
    ax2.legend()

    fig.suptitle(f'Sinusoidal wave  kx={kx:.3f} ky={ky:.3f}  T={T:.3f}', fontsize=13)
    fig.tight_layout()
    outfile = name + '_midlines.pdf'
    plt.savefig(outfile)
    print(f"Saved {outfile}")


def example_wave(nx_wave, ny_wave, petsc_args='', run=False):
    runwave(nx_wave, ny_wave, petsc_args=petsc_args, actually_run=run)


def add_wave_args(p):
    p.add_argument('--nx-wave', type=int, default=2,
                   help='integer wave number in x  (kx = nx_wave * 2pi/Lx, default 2)')
    p.add_argument('--ny-wave', type=int, default=0,
                   help='integer wave number in y  (ky = ny_wave * 2pi/Ly, default 0)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sinusoidal wave test for periodic BCs",
        usage="%(prog)s {run|plot} [--nx-wave N] [--ny-wave M]")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser  = subparsers.add_parser("run",  help="Run the simulation")
    plot_parser = subparsers.add_parser("plot", help="Generate midline plots")
    add_wave_args(run_parser)
    add_wave_args(plot_parser)

    args, unknown = parser.parse_known_args()

    if args.command == "run":
        example_wave(args.nx_wave, args.ny_wave, petsc_args=unknown, run=True)
    elif args.command == "plot":
        example_wave(args.nx_wave, args.ny_wave, petsc_args=unknown, run=False)
