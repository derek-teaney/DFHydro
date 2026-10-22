# Copilot Instructions for 2DFlow/DFHydro

## Project Overview
This is a C++17 relativistic viscous hydrodynamics simulation codebase for 2D fluid dynamics, implementing Bjorken expansion with Gaussian initial conditions. The project simulates superfluid dynamics using PETSc for numerical solving and HDF5 for data I/O.

## Architecture

### Core Components
- **DFHydro/**: Main simulation library (`Vischydro` class, `VischydroNode` data structures)
- **dtwork/**: Development workspace with solvers, tests, and experimental code  
- **example/**: Simple usage examples (`SimpleHydro.cpp` demonstrates basic workflow)
- **tests/**: Full simulation tests with Python integration (`VischydroMain.cpp`)

### Key Classes
- `Vischydro`: Main simulation class managing PETSc grid (`DM domain`), solution vectors (`Vec solution`), time stepping (`TS stepper`)
- `VischydroNode`: Per-grid-point data structure containing energy density `e`, velocity `u[2]`, stress tensor components
- `NonlinearSolver2D`: PETSc SNES-based solver for nonlinear equations (see `dtwork/test_solver.cpp`)

## Build System

### Dependencies
- **PETSc**: Required with MPI support. Use `pkg-config PETSc --cflags/--libs` for compilation flags
- **HDF5**: Optional but recommended. Enable with `-DPETSC_HAS_HDF5=1`
- **nlohmann/json**: For configuration files (header-only, in `extern/`)
- **mdspan**: C++23 multidimensional arrays (in `extern/mdspan/`)

### Build Process
```bash
mkdir build && cd build
cmake .. [-DPETSC_HAS_HDF5=1]  # Enable HDF5 if available
make
```

### Running Simulations
- Single-core example: `mpiexec -n 1 ./SimpleHydro.exe` (from build/example/)
- Multi-core solver: `mpiexec -n 4 ./VischydroMain.exe` (from build/tests/)
- Development testing: `mpiexec -n 1 ./test_solver -snes_monitor -snes_monitor_residual` (from build/dtwork/)

## Code Patterns

### Grid Access Pattern (Critical)
```cpp
// Get PETSc arrays for grid access
VischydroNode **asol;
DMDAVecGetArray(domain, vischydro.solution, &asol);
DMDACoor2d **xy;
DMDAVecGetArray(cdomain, vischydro.coordinates, &xy);

// Loop over local grid portion
PetscInt xs, ys, xm, ym;
DMDAGetCorners(domain, &xs, &ys, NULL, &xm, &ym, NULL);
for (PetscInt j = ys; j < ys + ym; j++) {
    for (PetscInt i = xs; i < xs + xm; i++) {
        // Access: asol[j][i].e, asol[j][i].u[0/1], xy[j][i].x/y
    }
}

// Always restore arrays
DMDAVecRestoreArray(domain, vischydro.solution, &asol);
DMDAVecRestoreArray(cdomain, vischydro.coordinates, &xy);
```

### Configuration Loading
- Use nlohmann::json for config files (see `dtwork/input.json`)
- Grid parameters: `nx`, `ny`, `xmin`, `xmax`, `ymin`, `ymax`
- Time settings: `t_start`, `t_end`, `dt`

### Data I/O Conventions
- HDF5 files: `{run_name}_grid.h5` for grid data, `{run_name}_ic.h5` for initial conditions
- ASCII files: `{run_name}_grid_t.txt` for timestep tracking
- Use `vischydro.save(filename)` and `vischydro.load_initial_conditions(filename)`

## Python Integration
- `tests/VischydroMainRun.py`: Main simulation runner with plotting capabilities
- `tests/vischydro.py`: Python utilities for data analysis
- Usage: `python VischydroMainRun.py run` then `python VischydroMainRun.py plot`

## Development Workflows

### Adding New Physics
1. Modify `VischydroNode` structure in `DFHydro/include/DFHydro/VischydroNode.hpp`
2. Update node filling logic in `DFHydro/src/VischydroNode.cpp`
3. Add corresponding test in `dtwork/` for validation

### Debugging PETSc Issues
- Use PETSc command line options: `-snes_monitor`, `-ksp_monitor`, `-log_view`
- Check `troubleshoot/` directory for standalone PETSc/MPI tests
- Verify MPI installation: `mpicxx` and `mpiexec` commands must match PETSc build

### Adding Tests
- Unit tests go in `dtwork/` (e.g., `test_*.cpp`)
- Integration tests go in `tests/` with corresponding Python runners
- Use `PetscCall()` macro for error checking in PETSc functions