#include <nlohmann/json.hpp>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#ifdef PETSC_HAVE_HDF5
#include <petscviewerhdf5.h>
#endif
#include "DFHydroMDSpan.hpp"
#include "Vischydro_impl.hpp"
#include <DFHydro/DFHydroEOS.hpp>
#include <DFHydro/Vischydro.hpp>
#include <petscerror.h>

namespace DFHydro {

using namespace DFHydro;

void Vischydro::load_initial_conditions(const std::string &filename,
                                        const std::string &initial_field_type) {
#ifdef PETSC_HAVE_HDF5
  PetscViewer viewer;
  PetscViewerHDF5Open(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_READ,
                      &viewer);
  PetscObjectSetName((PetscObject)solution, "initialdata");
  VecLoad(solution, viewer);
  PetscViewerDestroy(&viewer);

  // Loop over the elements of solution and call FillVischydroNode
  VischydroNode **asol;
  PetscCallVoid(DMDAVecGetArray(domain, solution, &asol));
  // Loop over grid points and calculate RHS
  PetscInt xs, ys, xm, ym;
  DMDAGetCorners(domain, &xs, &ys, NULL, &xm, &ym, NULL);

  for (PetscInt j = ys; j < ys + ym; j++) {
    for (PetscInt i = xs; i < xs + xm; i++) {
      if (initial_field_type == "charges") {
        // Use the conserved charges and find the primitive variables. Allow the
        // root finder to make an initial guess
        VischydroNode &n = asol[j][i];
        bool ok = vhnode_findstate(n, n.E, n.M[0], n.M[1], *eos);
        if (!ok) {
          std::cout << "Initialization Error: Root finding failed at " << i
                    << ", " << j << std::endl;
          std::abort();
        }
      } else if (initial_field_type == "landau") {
        // Interpret the file contents as Landau-frame primitives plus viscous
        // stresses, then convert them to the density-frame state used by the
        // solver.
        VischydroNode nLF = asol[j][i];
        vhnode_fill_LF(nLF, *eos);

        VischydroNode nDF;
        bool ok = vhnode_LFtoDF(*eos, nLF, nDF);
        if (!ok) {
          std::cout << "Initialization Error: Landau->density conversion failed at "
                    << i << ", " << j << std::endl;
          std::abort();
        }
        asol[j][i] = nDF;
      } else {
        vhnode_fill(asol[j][i], *eos);
      }
    }
  }
  // Return the pointer to the local array back to the memory space
  PetscCallVoid(DMDAVecRestoreArray(domain, solution, &asol));

#else
  PetscPrintf(PETSC_COMM_WORLD, "HDF5 support not available. Cannot load "
                                "initial conditions from file.\n");
#endif
}

// Save the current grid to a file using HDF5. The filename is optional and
// defaults to output.h5
void Vischydro::save(const std::string &filename) {
#ifdef PETSC_HAVE_HDF5
  PetscViewer viewer;
  PetscViewerHDF5Open(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_WRITE,
                      &viewer);
  PetscObjectSetName((PetscObject)solution, "output");
  VecView(solution, viewer);

  PetscObjectSetName((PetscObject)coordinates, "coordinates");
  VecView(coordinates, viewer);
  PetscViewerDestroy(&viewer);
#else
  PetscPrintf(PETSC_COMM_WORLD,
              "HDF5 support not available. Cannot save to file.\n");
#endif
}
void Vischydro::solve(double t1, double t2, double dt) {
  PetscCallVoid(TSSetTime(stepper, t1));
  PetscCallVoid(TSSetMaxTime(stepper, t2));
  PetscCallVoid(TSSetTimeStep(stepper, dt));
  PetscCallVoid(TSSolve(stepper, qsolution));
}

PetscErrorCode TransferSolutionToQGrid(const DM &domain, const Vec &solution,
                                       const DM &qdomain,
                                       const Vec &qsolution) {
  VischydroNode **asol;
  PetscCall(DMDAVecGetArrayRead(domain, solution, &asol));

  VischydroQNode **qsol;
  PetscCall(DMDAVecGetArray(qdomain, qsolution, &qsol));

  PetscInt ixs, iys, ixm, iym;
  DMDAGetCorners(domain, &ixs, &iys, NULL, &ixm, &iym, NULL);
  for (PetscInt j = iys; j < iys + iym; j++) {
    for (PetscInt i = ixs; i < ixs + ixm; i++) {
      qsol[j][i].E = asol[j][i].E;
      qsol[j][i].M[0] = asol[j][i].M[0];
      qsol[j][i].M[1] = asol[j][i].M[1];
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(domain, solution, &asol));
  PetscCall(DMDAVecRestoreArray(qdomain, qsolution, &qsol));
  return 0;
}
PetscErrorCode TransferQGridToSolution(const DM &qdomain, const Vec &qsolution,
                                       const DM &domain, const Vec &solution) {
  VischydroNode **asol;
  PetscCall(DMDAVecGetArray(domain, solution, &asol));

  VischydroQNode **qsol;
  PetscCall(DMDAVecGetArrayRead(qdomain, qsolution, &qsol));

  PetscInt ixs, iys, ixm, iym;
  DMDAGetCorners(domain, &ixs, &iys, NULL, &ixm, &iym, NULL);
  for (PetscInt j = iys; j < iys + iym; j++) {
    for (PetscInt i = ixs; i < ixs + ixm; i++) {
      asol[j][i].E = qsol[j][i].E;
      asol[j][i].M[0] = qsol[j][i].M[0];
      asol[j][i].M[1] = qsol[j][i].M[1];
    }
  }
  PetscCall(DMDAVecRestoreArray(domain, solution, &asol));
  PetscCall(DMDAVecRestoreArrayRead(qdomain, qsolution, &qsol));
  return 0;
}

// Returns the largest and smalllest (most-negative) propagation velocities for
// a given speed of sound cs2, velocity ux, and Lorentz factor u0.
std::tuple<double, double> idealPropagationVelocity(const double &cs2,
                                                    const double &ux,
                                                    const double &u0) {
  double ut = u0;
  double uk = ux;
  const double A = ut * uk * (1. - cs2);
  const double B = (ut * ut - uk * uk - (ut * ut - uk * uk - 1.) * cs2) * cs2;
  const double D = ut * ut * (1. - cs2) + cs2;
  double ap = (A + sqrt(B)) / D;
  double am = (A - sqrt(B)) / D;
  return std::make_tuple(ap, am);
}

// Given two states, left and right, this function returns the largest and
// smallest propagation velocities, ap and am, respectively. The states are
// given by the speed of sound cs2 and the velocity ux and Lorentz factor u0. If
// usespeedoflight is true, then the propagation velocities are set to 1.01 and
// -1.01, respectively.
std::tuple<double, double>
propagationVelocity(const double &cs2L, const double &uxL, const double &u0L,
                    const double &cs2R, const double &uxR, const double &u0R,
                    bool usespeedoflight = false) {
  double ap, am;
  if (usespeedoflight) {
    ap = 1.01;
    am = -1.01;

  } else {
    auto [apl, aml] = idealPropagationVelocity(cs2L, uxL, u0L);
    auto [apr, amr] = idealPropagationVelocity(cs2R, uxR, u0R);
    ap = std::max(std::max(apl, apr), 0.0);
    am = std::min(std::min(aml, amr), 0.0);
    if (std::abs(ap) > 1.0 || std::abs(am) > 1.0) {
      std::cout << "**propagationVelocity*** superluminal velocity!"
                << std::endl;
      std::cout << ap << " " << am << std::endl;
      std::abort();
    }
  }
  return std::make_tuple(ap, am);
}

void findstate_problem(const std::string &context, const PetscInt &i, const PetscInt &j,
                       const VischydroNode &n_last, VischydroNode &n,
                       const EOS &eos) {
  static int count = 0;
  std::cout << "findstate_problem in context: " << context << " at (" << j
            << ", " << i << ")" << std::endl;
  std::cout << "Input energy density: " << n_last.e << std::endl;
  n.print("VischydroNode state:");
  count++;
  const int max_errors = 500;
  std::cout << "Reverting to last known good state. Error count = " << count
            << " / " << max_errors << std::endl;
  n = n_last;
  if (count > max_errors) {
    std::cout << "Too many findstate problems, aborting." << std::endl;
    std::abort();
  }
}

void set_boundary_conditions(VischydroNode **asol, const DMDALocalInfo &info,
                             bool is_periodic) {

  if (is_periodic == true) {
    return; // Boundary conditions are handled automatically by PETSc DMDA
  }

  PetscInt ixs = info.xs;
  PetscInt jys = info.ys;
  PetscInt ixm = info.xm;
  PetscInt jym = info.ym;
  PetscInt mx = info.mx;
  PetscInt my = info.my;
  int nbc = 2;

  // Set boundary conditions in x direction
  for (PetscInt j = jys - nbc; j < jys + jym + nbc; j++) {
    for (PetscInt n = 1; n <= nbc; n++) {
      if (ixs == 0) {
        // Left boundary
        PetscInt i_left = ixs - n;
        PetscInt i_src, j_src;
        i_src = ixs;
        j_src = std::min(std::max(j, PetscInt{0}), my - 1);
        asol[j][i_left] = asol[j_src][i_src];
      }
      if (ixs + ixm == mx) {
        // Right boundary
        PetscInt i_right = ixs + ixm - 1 + n;
        PetscInt i_src, j_src;
        i_src = ixs + ixm - 1;
        j_src = std::min(std::max(j, PetscInt{0}), my - 1);
        asol[j][i_right] = asol[j_src][i_src];
      }
    }
  }
  // Set boundary conditions in y direction
  for (PetscInt i = ixs - nbc; i < ixs + ixm + nbc; i++) {
    for (PetscInt n = 1; n <= nbc; n++) {
      // Bottom boundary
      if (jys == 0) {
        PetscInt j_bottom = jys - n;
        PetscInt j_src, i_src;
        j_src = jys;
        i_src = std::min(std::max(i, PetscInt{0}), mx - 1);
        asol[j_bottom][i] = asol[j_src][i_src];
      }
      // Top boundary
      if (jys + jym == my) {
        PetscInt j_top = jys + jym - 1 + n;
        PetscInt j_src = jys + jym - 1;
        PetscInt i_src = std::min(std::max(i, PetscInt{0}), mx - 1);
        asol[j_top][i] = asol[j_src][i_src];
      }
    }
  }
}

// Compute the right-hand side function for the Euler equations using the KT
// scheme. The function G is computed from the current state U (a vector of the
// conserved variables on the charge grid) and G (also a vector on the charge
// grid) holds dQ/dt at time t. The context ctx contains a pointer to the
// Vischydro object that holds the simulation parameters and state.
PetscErrorCode EulerRHSFunction(TS ts, PetscReal t, Vec U, Vec G, void *ctx) {

  PetscLogEvent event;
  PetscLogEventRegister("EulerRHSFunction", 0, &event);
  PetscLogEventBegin(event, 0, 0, 0, 0);

  const Vischydro &run = *(Vischydro *)ctx;
  auto &eos = *run.eos;

  PetscCall(TransferQGridToSolution(run.qdomain, U, run.domain, run.solution));

  // Zero out the rhs vector and get pointer to local array
  VecZeroEntries(G);
  VischydroQNode **ag;
  PetscCall(DMDAVecGetArray(run.qdomain, G, &ag));

  // Copy the global solution to local solutions including the boundary values
  PetscCall(DMGlobalToLocal(run.domain, run.solution, INSERT_VALUES,
                            run.local_solution));
  VischydroNode **asol;
  PetscCall(DMDAVecGetArray(run.domain, run.local_solution, &asol));

  // Get the grid information
  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(run.domain, &info));

  // Set the boundary condtions
  set_boundary_conditions(asol, info, run.has_periodic_bc());

  // Loop over grid points and calculate RHS
  PetscInt ixs, jys, ixm, jym;
  DMDAGetCorners(run.domain, &ixs, &jys, NULL, &ixm, &jym, NULL);

  const double epsilon = 1.e-8;
  limitter slope(limitter::kCenteredMinMod);

  for (PetscInt j = jys - 2; j < jys + jym + 2; j++) {
    for (PetscInt i = ixs - 2; i < ixs + ixm + 2; i++) {
      bool ok = vhnode_findstate(asol[j][i].e, asol[j][i], eos);
      if (!ok) {
        findstate_problem("EulerRHSFunction init", i, j, asol[j][i], asol[j][i],
                          eos);
      }
    }
  }

  for (PetscInt j = jys; j < jys + jym; j++) {
    for (PetscInt i = ixs; i < ixs + ixm + 1; i++) {
      VischydroNode nL{};
      VischydroNode nR{};

      // extrapolate i-1 to i-1/2
      {
        VischydroNode &np = asol[j][i];
        VischydroNode &n = asol[j][i - 1];
        VischydroNode &nm = asol[j][i - 2];
        nL.e = n.e + 0.5 * slope(nm.e, n.e, np.e);
        nL.u[0] = n.u[0] + 0.5 * slope(nm.u[0], n.u[0], np.u[0]);
        nL.u[1] = n.u[1] + 0.5 * slope(nm.u[1], n.u[1], np.u[1]);
        vhnode_fill(nL, eos);
      }

      // extrapolate i to i-1/2
      {
        VischydroNode &np = asol[j][i + 1];
        VischydroNode &n = asol[j][i];
        VischydroNode &nm = asol[j][i - 1];
        nR.e = n.e - 0.5 * slope(nm.e, n.e, np.e);
        nR.u[0] = n.u[0] - 0.5 * slope(nm.u[0], n.u[0], np.u[0]);
        nR.u[1] = n.u[1] - 0.5 * slope(nm.u[1], n.u[1], np.u[1]);
        vhnode_fill(nR, eos);
      }

      // Compute the mean flux
      auto FL = nL.fluxX();
      auto FR = nR.fluxX();
      auto qL = nL.charge();
      auto qR = nR.charge();

      // Compute the wave spreads and use this to determine the flux
      auto [lambdap, lambdam] = propagationVelocity(nL.cs2, nL.u[0], nL.u0(),
                                                    nR.cs2, nR.u[0], nR.u0());

      // Compute the wave spreads and use this to determine the flux
      double ap = std::max(epsilon, lambdap);
      double am = std::max(epsilon, -lambdam);

      std::array<double, VischydroNode::Ncharge> F{};
      for (int k = 0; k < VischydroNode::Ncharge; k++) {
        F[k] =
            (ap * FL[k] + am * FR[k] - ap * am * (qR[k] - qL[k])) / (ap + am);
      }
      if (i > ixs) {
        ag[j][i - 1].E -= F[0] / run.get_dx();
        ag[j][i - 1].M[0] -= F[1] / run.get_dx();
        ag[j][i - 1].M[1] -= F[2] / run.get_dx();
      }
      if (i < ixs + ixm) {
        ag[j][i].E += F[0] / run.get_dx();
        ag[j][i].M[0] += F[1] / run.get_dx();
        ag[j][i].M[1] += F[2] / run.get_dx();
      }
    }
  }
  for (PetscInt i = ixs; i < ixs + ixm; i++) {
    for (PetscInt j = jys; j < jys + jym + 1; j++) {
      VischydroNode nL{};
      VischydroNode nR{};

      // extrapolate j-1 to j-1/2
      {
        VischydroNode &np = asol[j][i];
        VischydroNode &n = asol[j - 1][i];
        VischydroNode &nm = asol[j - 2][i];
        nL.e = n.e + 0.5 * slope(nm.e, n.e, np.e);
        nL.u[0] = n.u[0] + 0.5 * slope(nm.u[0], n.u[0], np.u[0]);
        nL.u[1] = n.u[1] + 0.5 * slope(nm.u[1], n.u[1], np.u[1]);
        vhnode_fill(nL, eos);
      }

      // extrapolate j to j-1/2
      {
        VischydroNode &np = asol[j + 1][i];
        VischydroNode &n = asol[j][i];
        VischydroNode &nm = asol[j - 1][i];
        nR.e = n.e - 0.5 * slope(nm.e, n.e, np.e);
        nR.u[0] = n.u[0] - 0.5 * slope(nm.u[0], n.u[0], np.u[0]);
        nR.u[1] = n.u[1] - 0.5 * slope(nm.u[1], n.u[1], np.u[1]);
        vhnode_fill(nR, eos);
      }

      // Compute the mean flux
      auto FL = nL.fluxY();
      auto FR = nR.fluxY();
      auto qL = nL.charge();
      auto qR = nR.charge();

      // Compute the wave spreads and use this to determine the flux
      auto [lambdap, lambdam] = propagationVelocity(nL.cs2, nL.u[1], nL.u0(),
                                                    nR.cs2, nR.u[1], nR.u0());

      // Compute the wave spreads and use this to determine the flux
      double ap = std::max(epsilon, lambdap);
      double am = std::max(epsilon, -lambdam);

      std::array<double, VischydroNode::Ncharge> F{};
      for (int k = 0; k < VischydroNode::Ncharge; k++) {
        F[k] =
            (ap * FL[k] + am * FR[k] - ap * am * (qR[k] - qL[k])) / (ap + am);
      }
      if (j > jys) {
        ag[j - 1][i].E -= F[0] / run.get_dy();
        ag[j - 1][i].M[0] -= F[1] / run.get_dy();
        ag[j - 1][i].M[1] -= F[2] / run.get_dy();
      }
      if (j < jys + jym) {
        ag[j][i].E += F[0] / run.get_dy();
        ag[j][i].M[0] += F[1] / run.get_dy();
        ag[j][i].M[1] += F[2] / run.get_dy();
      }
    }
  }

  // Handle Bjorken expansion source terms
  if (run.is_bjorken_expansion()) {
    for (PetscInt j = jys; j < jys + jym; j++) {
      for (PetscInt i = ixs; i < ixs + ixm; i++) {
        double tau = t;
        if (tau > 0.0) {
          ag[j][i].E += -(asol[j][i].E + asol[j][i].p) / tau;
          ag[j][i].M[0] += -asol[j][i].M[0] / tau;
          ag[j][i].M[1] += -asol[j][i].M[1] / tau;
        }
      }
    }
  }

  // Return the pointer to the local array back to the memory space
  // PetscCall(
  //     DMDAVecRestoreArray(run.domain, run.local_solution_last, &asol_last));
  PetscCall(DMDAVecRestoreArray(run.domain, run.local_solution, &asol));
  PetscCall(DMDAVecRestoreArray(run.qdomain, G, &ag));
  PetscLogEventEnd(event, 0, 0, 0, 0);

  return 0;
}

// PreStep function to transfer solution from solution vector to qsolution which
// is used by the TS solver.
PetscErrorCode PreStep(TS ts) {

  Vischydro *run = nullptr;
  TSGetApplicationContext(ts, &run);

  Vec Y = nullptr;
  TSGetSolution(ts, &Y);
  TransferSolutionToQGrid(run->domain, run->solution, run->qdomain, Y);
  return 0;
}

// Evaluate the inverse susceptibility matrix at a vertex given the four
// surrounding nodes
void evaluate_vertex_chiinv(const VischydroNode &n00, const VischydroNode &n01,
                            const VischydroNode &n10, const VischydroNode &n11,
                            const EOS &eos, std::array<double, 36> &chiinv_d) {

  MDSpan<double, 4, 9> chiinv(chiinv_d.data());
  // Create an array of node references for the four surrounding nodes
  std::array<std::reference_wrapper<const VischydroNode>, 4> nds{n00, n01, n10,
                                                                 n11};

  for (int s = 0; s < 4; s++) {
    std::array<double, 9> chiinvs{};
    vhnode_chiinv(nds[s], eos, chiinvs);
    for (int a = 0; a < 9; a++) {
      chiinv(s, a) = chiinvs[a];
    }
  }
  return;
}

// Evaluate kappa at a vertex given the four surrounding nodes
void evaluate_vertex_k(const VischydroNode &n00, const VischydroNode &n01,
                       const VischydroNode &n10, const VischydroNode &n11,
                       const EOS &eos, double &knn,
                       std::array<double, 4> &knx_d,
                       std::array<double, 16> &kxx_d) {
  // Average state at the vertex
  VischydroNode nv{};
  nv.E = 0.25 * (n00.E + n01.E + n10.E + n11.E);
  nv.M[0] = 0.25 * (n00.M[0] + n01.M[0] + n10.M[0] + n11.M[0]);
  nv.M[1] = 0.25 * (n00.M[1] + n01.M[1] + n10.M[1] + n11.M[1]);
  double e = 0.25 * (n00.e + n01.e + n10.e + n11.e);
  bool ok = vhnode_findstate(e, nv, eos);
  if (!ok) {
    nv.e = e;
    findstate_problem("evaluate_vertex_k", -1, -1, nv, nv, eos);
  }
  vhnode_kappa(nv, eos, knn, knx_d, kxx_d);
}

// Returns the derivatave matrix used in the finite difference calculations for
// the vertex-centered quantities. Here dx and dy are the grid spacings in the x
// and y directions, respectively.
void get_derivative_matrix(const double &dx, const double &dy,
                           std::array<double, 8> &Dx_d) {
  double tdx = 2 * dx;
  double tdy = 2 * dy;
  // Derivative matrices
  Dx_d = {-1. / tdx, -1. / tdy, 1. / tdx, -1. / tdy,
          -1. / tdx, 1. / tdy,  1. / tdx, 1. / tdy};
  return;
}

enum class ViscousFunctionType { Difference, Average };

PetscErrorCode ViscousFunction(TS ts, PetscReal t, Vec U, Vec F, void *context,
                               enum ViscousFunctionType type) {

  PetscLogEvent event;
  PetscLogEventRegister("ViscousFunction", 0, &event);
  PetscLogEventBegin(event, 0, 0, 0, 0);

  auto run = (Vischydro *)context;

  // Transfer the solution vector u, which is on the q-grid to the solution grid
  TransferQGridToSolution(run->qdomain, U, run->domain, run->solution);

  // Do communcation and fill up boundary cells fillin local_solution based on u
  PetscCall(DMGlobalToLocal(run->domain, run->solution, INSERT_VALUES,
                            run->local_solution));

  // Local array with the boundary cells
  VischydroNode **au;
  PetscCall(DMDAVecGetArray(run->domain, run->local_solution, &au));

  // This is the vector that holds the LHS function,  differences of strains
  VischydroQNode **aF;
  if (type == ViscousFunctionType::Difference) {
    PetscCall(DMDAVecGetArray(run->qdomain, F, &aF));
  }
  VischydroNode **aU = nullptr;
  if (type == ViscousFunctionType::Average) {
    PetscCall(DMDAVecGetArray(run->domain, run->solution, &aU));
  }

  // Get the grid information
  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(run->domain, &info));
  set_boundary_conditions(au, info, run->has_periodic_bc());

  PetscInt ixs, ixm, jys, jym;
  DMDAGetCorners(run->domain, &ixs, &jys, 0, &ixm, &jym, 0);

  // Loop over the grid and call findstate to ensure primitives are recovered
  for (PetscInt j = jys - 1; j < jys + jym + 1; j++) {
    for (PetscInt i = ixs - 1; i < ixs + ixm + 1; i++) {
      bool ok = vhnode_findstate(au[j][i].e, au[j][i], *run->eos);
      if (!ok) {
        findstate_problem("ViscousFunction", i, j, au[j][i], au[j][i],
                          *run->eos);
      }
      if (type == ViscousFunctionType::Average) {
        // check that we are in the computational domain
        if (i >= ixs and i < ixs + ixm and j >= jys and j < jys + jym) {
          aU[j][i] = au[j][i];
          aU[j][i].set_viscous_stress(0.0, 0.0, 0.0,
                                      0.0); // reset viscous stresses
        }
      }
    }
  }

  double gtt = -1.0;
  double k33 = (run->is_bjorken_expansion() ? gtt / (4 * t) : 0);
  double k33average = (run->is_bjorken_expansion() ? 1.0 / 4.0 : 0);
  std::array<double, 4> K{k33, k33, k33, k33};

  std::array<double, 8> Dx_d{};
  MDSpan<double, 4, 2> Dx(Dx_d.data());
  get_derivative_matrix(run->get_dx(), run->get_dy(), Dx_d);

  std::array<double, 4> Bt{};
  std::array<double, 8> Bx_d{};
  MDSpan<double, 4, 2> Bx(Bx_d.data());

  double knn = 0.0;
  std::array<double, 4> knx_d{};
  MDSpan<double, 2, 2> knx(knx_d.data());
  std::array<double, 16> kxx_d{};
  MDSpan<double, 2, 2, 2, 2> kxx(kxx_d.data());

  std::array<int, 4> is = {0, 1, 0, 1};
  std::array<int, 4> js = {0, 0, 1, 1};

  for (PetscInt j = jys - 1; j < jys + jym; j++) {
    for (PetscInt i = ixs - 1; i < ixs + ixm; i++) {

      Bt = {gtt * au[j][i].b0(), gtt * au[j][i + 1].b0(),
            gtt * au[j + 1][i].b0(), gtt * au[j + 1][i + 1].b0()};

      // clang-format off
      Bx_d = {au[j][i].bx(), au[j][i].by(),
              au[j][i + 1].bx(), au[j][i + 1].by(),
              au[j + 1][i].bx(), au[j + 1][i].by(),
              au[j + 1][i + 1].bx(), au[j + 1][i + 1].by()};
      // clang-format on

      evaluate_vertex_k(au[j][i], au[j][i + 1], au[j + 1][i], au[j + 1][i + 1],
                        *run->eos, knn, knx_d, kxx_d);

      for (int s = 0; s < 4; s++) {
        int ip = i + is[s];
        int jp = j + js[s];

        // skip if outside computational domain
        if (not(ip >= ixs and ip < ixs + ixm and jp >= jys and
                jp < jys + jym)) {
          continue;
        }

        for (int sp = 0; sp < 4; sp++) {
          PetscScalar pinn = knn * K[sp] * Bt[sp] +
                             knx(0, 0) * Dx(sp, 0) * Bx(sp, 0) +
                             knx(0, 1) * Dx(sp, 0) * Bx(sp, 1) +
                             knx(1, 0) * Dx(sp, 1) * Bx(sp, 0) +
                             knx(1, 1) * Dx(sp, 1) * Bx(sp, 1);

          switch (type) {
          case ViscousFunctionType::Difference:
            aF[jp][ip].E += K[s] * pinn;
            break;
          case ViscousFunctionType::Average:
            aU[jp][ip].get_pinn() += -k33average * pinn;
            break;
          }

          for (int l1 = 0; l1 < 2; l1++) {
            for (int l2 = 0; l2 < 2; l2++) {
              PetscScalar piij = knx(l1, l2) * K[sp] * Bt[sp] +
                                 kxx(l1, l2, 0, 0) * Dx(sp, 0) * Bx(sp, 0) +
                                 kxx(l1, l2, 0, 1) * Dx(sp, 0) * Bx(sp, 1) +
                                 kxx(l1, l2, 1, 0) * Dx(sp, 1) * Bx(sp, 0) +
                                 kxx(l1, l2, 1, 1) * Dx(sp, 1) * Bx(sp, 1);

              switch (type) {
              case ViscousFunctionType::Difference:
                aF[jp][ip].M[l1] += Dx(s, l2) * piij;
                break;
              case ViscousFunctionType::Average:
                aU[jp][ip].get_piij(l1, l2) += -0.25 * piij;
                break;
              }
            }
          }
        }
      }
    }
  }
  if (type == ViscousFunctionType::Difference) {
    PetscCall(DMDAVecRestoreArray(run->qdomain, F, &aF));
  } else {
    PetscCall(DMDAVecRestoreArray(run->domain, run->solution, &aU));
  }
  PetscCall(DMDAVecRestoreArray(run->domain, run->local_solution, &au));
  PetscLogEventEnd(event, 0, 0, 0, 0);
  return 0;
}

// LHS implicit function for the viscous hydro equations. See notes for
// expalanation.
PetscErrorCode LHSIFunction2(TS ts, PetscReal t, Vec u, Vec udot, Vec F,
                             void *context) {
  // This is just copying udot to F. F is updated below
  VecCopy(udot, F);
  return ViscousFunction(ts, t, u, F, context, ViscousFunctionType::Difference);
}

PetscErrorCode LHSIJacobian2(TS ts, PetscReal t, Vec u, Vec udot,
                             PetscReal shift, Mat J, Mat P, void *context) {

  PetscLogEvent event;
  PetscLogEventRegister("LHSIJacobian2", 0, &event);
  PetscLogEventBegin(event, 0, 0, 0, 0);

  auto run = (Vischydro *)context;

  TransferQGridToSolution(run->qdomain, u, run->domain, run->solution);

  // Do communcation and fill up boundary cells fillin local_solution based on u
  PetscCall(DMGlobalToLocal(run->domain, run->solution, INSERT_VALUES,
                            run->local_solution));

  // Local array with the boundary cells
  VischydroNode **au;
  PetscCall(DMDAVecGetArray(run->domain, run->local_solution, &au));

  // Get the grid information
  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(run->domain, &info));
  set_boundary_conditions(au, info, run->has_periodic_bc());

  PetscInt ixs, ixm, jys, jym;
  DMDAGetCorners(run->domain, &ixs, &jys, 0, &ixm, &jym, 0);

  // Loop over the grid and call idealHydroCellSolve
  for (PetscInt j = jys - 1; j < jys + jym + 1; j++) {
    for (PetscInt i = ixs - 1; i < ixs + ixm + 1; i++) {
      bool ok = vhnode_findstate(au[j][i].e, au[j][i], *run->eos);
      if (!ok) {
        findstate_problem("LHSIJacobian2", i, j, au[j][i], au[j][i], *run->eos);
      }
    }
  }

  double dx = run->get_dx();
  double dy = run->get_dy();

  // Is this needed?
  PetscCall(MatZeroEntries(P));
  PetscCall(MatShift(P, shift));

  MatStencil row{};
  std::array<MatStencil, 12> columns_d{};
  std::array<PetscScalar, 12> values_d{};

  std::array<PetscScalar, 36> chiinv_d{};
  MDSpan<PetscScalar, 4, 3, 3> chiinv(chiinv_d.data());

  double gtt = -1.0;
  double k33 = (run->get_highest_order_term_only() ? 0. : 1.) *
               (run->is_bjorken_expansion() ? gtt / (4 * t) : 0);

  std::array<double, 4> K{k33, k33, k33, k33};
  std::array<double, 8> Dx_d{};
  MDSpan<double, 4, 2> Dx(Dx_d.data());
  get_derivative_matrix(dx, dy, Dx_d);

  double knn{};
  std::array<double, 4> knx_d{};
  MDSpan<double, 2, 2> knx(knx_d.data());
  std::array<double, 16> kxx_d{};
  MDSpan<double, 2, 2, 2, 2> kxx(kxx_d.data());

  std::array<int, 4> is = {0, 1, 0, 1};
  std::array<int, 4> js = {0, 0, 1, 1};

  for (PetscInt j = jys - 1; j < jys + jym; j++) {
    for (PetscInt i = ixs - 1; i < ixs + ixm; i++) {

      // This call is 10% of the event loop
      evaluate_vertex_k(au[j][i], au[j][i + 1], au[j + 1][i], au[j + 1][i + 1],
                        *run->eos, knn, knx_d, kxx_d);

      evaluate_vertex_chiinv(au[j][i], au[j][i + 1], au[j + 1][i],
                             au[j + 1][i + 1], *run->eos, chiinv_d);

      for (int s1 = 0; s1 < 4; s1++) {
        for (int c1 = 0; c1 < 3; c1++) {

          PetscInt ip1 = i + is[s1];
          PetscInt jp1 = j + js[s1];

          // skip if the row is outside computational domain.
          if (not(ip1 >= ixs and ip1 < ixs + ixm and jp1 >= jys and
                  jp1 < jys + jym)) {
            continue;
          }

          row.i = ip1; // row(s1, c1);
          row.j = jp1;
          row.c = c1;
          if (c1 == 0 and run->get_highest_order_term_only()) {
            continue;
          }

          int nc = 0;
          int nv = 0;
          for (int s2 = 0; s2 < 4; s2++) {
            for (int c2 = 0; c2 < 3; c2++) {

              PetscInt ip2 = i + is[s2];
              PetscInt jp2 = j + js[s2];

              if (not run->has_periodic_bc()) {
                ip2 = std::min(std::max(ip2, PetscInt{0}), info.mx - 1);
                jp2 = std::min(std::max(jp2, PetscInt{0}), info.my - 1);
              }

              auto &column = columns_d[nc++]; // columns(s2, c2);
              column.i = ip2;
              column.j = jp2;
              column.c = c2;

              if (c1 == 0) {
                auto &value = values_d[nv++]; // values(s1, c1, s2, c2);
                value = K[s1] * knn * K[s2] * chiinv(s2, c2, 0) +
                        K[s1] * knx(0, 0) * Dx(s2, 0) * chiinv(s2, c2, 1 + 0) +
                        K[s1] * knx(0, 1) * Dx(s2, 0) * chiinv(s2, c2, 1 + 1) +
                        K[s1] * knx(1, 0) * Dx(s2, 1) * chiinv(s2, c2, 1 + 0) +
                        K[s1] * knx(1, 1) * Dx(s2, 1) * chiinv(s2, c2, 1 + 1);
              } else if (c1 == 1 or c1 == 2) {
                int l1 = c1 - 1;
                auto &value = values_d[nv++]; // values(s1, c1, s2, c2);
                value = 0.0;
                for (int l2 = 0; l2 < 2; l2++) {
                  value +=
                      Dx(s1, l2) * knx(l1, l2) * K[s2] * chiinv(s2, c2, 0) +
                      Dx(s1, l2) * kxx(l1, l2, 0, 0) * Dx(s2, 0) *
                          chiinv(s2, c2, 1 + 0) +
                      Dx(s1, l2) * kxx(l1, l2, 0, 1) * Dx(s2, 0) *
                          chiinv(s2, c2, 1 + 1) +
                      Dx(s1, l2) * kxx(l1, l2, 1, 0) * Dx(s2, 1) *
                          chiinv(s2, c2, 1 + 0) +
                      Dx(s1, l2) * kxx(l1, l2, 1, 1) * Dx(s2, 1) *
                          chiinv(s2, c2, 1 + 1);
                }
              }
            }
          }
          MatSetValuesStencil(P, 1, &row, nc, columns_d.data(), values_d.data(),
                              ADD_VALUES);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(run->domain, run->local_solution, &au));
  MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
  if (J != P) {
    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
  }
  PetscLogEventEnd(event, 0, 0, 0, 0);

  return 0;
}

// PostStep function to transfer solution from qsolution used by the TS solver
// back to solution vector. This happens after each time step. Then the
// primitives are recovered in the solution vector.
PetscErrorCode PostStepInversion(TS ts) {
  Vischydro *runptr = nullptr;
  TSGetApplicationContext(ts, &runptr);

  Vec u = nullptr;
  TSGetSolution(ts, &u);

  PetscScalar t;
  TSGetTime(ts, &t);

  return ViscousFunction(ts, t, u, 0, runptr, ViscousFunctionType::Average);
}

// PostStage function to transfer solution from qsolution used by the TS solver
// back to solution vector. This happens after each stage of the solver. Then
// the primitives are recovered in the solution vector.
PetscErrorCode PostStageInversion(TS ts, PetscReal stagetime,
                                  PetscInt stageindex, Vec *Y) {
  Vischydro *runptr = nullptr;
  TSGetApplicationContext(ts, &runptr);
  Vischydro &run = *runptr;

  TransferQGridToSolution(run.qdomain, Y[stageindex], run.domain, run.solution);

  // Recover the primitives
  VischydroNode **au;
  PetscCall(DMDAVecGetArray(run.domain, run.solution, &au));

  PetscInt ixs, ixm, jys, jym;
  DMDAGetCorners(run.domain, &ixs, &jys, NULL, &ixm, &jym, NULL);
  for (PetscInt j = jys; j < jys + jym; j++) {
    for (PetscInt i = ixs; i < ixs + ixm; i++) {
      bool ok = vhnode_findstate(au[j][i].e, au[j][i], *run.eos);
      if (!ok) {
        std::string context =
            "PostStageInversion_" + std::to_string(stageindex);
        findstate_problem(context, i, j, au[j][i], au[j][i], *run.eos);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(run.domain, run.solution, &au));
  return 0;
}

Vischydro::Vischydro(const int &nx_i, const double &xmin_i,
                     const double &xmax_i, const int &ny_i,
                     const double &ymin_i, const double &ymax_i,
                     const EOS *eos_i, const nlohmann::json &config)
    : eos(eos_i), nx(nx_i), xmin(xmin_i), xmax(xmax_i), ny(ny_i), ymin(ymin_i),
      ymax(ymax_i) {

  defaults.update(config);

  cfl = defaults.at("cfl_max").get<double>();
  is_bjorken = defaults.at("is_bjorken").get<bool>();
  is_periodic = defaults.at("is_periodic").get<bool>();
  use_ideal_step_only = defaults.at("use_ideal_step_only").get<bool>();
  highest_order_term_only = defaults.at("highest_order_term_only").get<bool>();
  double Lx = xmax - xmin;
  double Ly = ymax - ymin;

  dx = Lx / nx;
  dy = Ly / ny;

  const int stencil_width = 2;
  // 2d grid with periodic boundary conditions
  if (is_periodic) {
    DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
                 DMDA_STENCIL_BOX, nx, ny, PETSC_DECIDE, PETSC_DECIDE,
                 VischydroNode::NDOF, stencil_width, NULL, NULL, &domain);
  } else {
    DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                 DMDA_STENCIL_BOX, nx, ny, PETSC_DECIDE, PETSC_DECIDE,
                 VischydroNode::NDOF, stencil_width, NULL, NULL, &domain);
  }
  DMSetFromOptions(domain);
  DMSetUp(domain);
  DMCreateGlobalVector(domain, &solution);
  DMCreateLocalVector(domain, &local_solution);

  // Create a domain and vector for the charges only. This is what the time
  // stepper will work on.
  DMDACreateCompatibleDMDA(domain, VischydroNode::Ncharge, &qdomain);
  DMCreateGlobalVector(qdomain, &qsolution);

  // Set coordinates
  if (is_periodic) {
    DMDASetUniformCoordinates(domain, xmin, xmax, ymin, ymax, 0.0, 0.0);
  } else {
    DMDASetUniformCoordinates(domain, xmin, xmax - dx, ymin, ymax - dy, 0.0,
                              0.0);
  }
  DMGetCoordinates(domain, &coordinates);
  DMGetCoordinateDM(domain, &cdomain);

  // Create the time stepper object of PETSc.
  // We note that the time stepper will work on the qdomain and qsolution
  // which only contains the charges needed for the hydrodynamics evolution.
  TSCreate(PETSC_COMM_WORLD, &stepper);
  TSSetApplicationContext(stepper, this);
  TSSetDM(stepper, qdomain);
  TSSetSolution(stepper, qsolution);

  // Simplest possible time step adaptivity: no adaptivity
  TSSetTimeStep(stepper, get_default_time_step());
  TSAdapt adapt;
  TSGetAdapt(stepper, &adapt);
  TSAdaptSetType(adapt, TSADAPTNONE);
  TSSetExactFinalTime(stepper, TS_EXACTFINALTIME_MATCHSTEP);

  // Set the hooks for pre-step, post-step, and post-stage
  TSSetPreStep(stepper, PreStep);
  TSSetPostStep(stepper, PostStepInversion);
  TSSetPostStage(stepper, PostStageInversion);

  if (use_ideal_step_only) {
    TSSetType(stepper, TSSSP);
    TSSetRHSFunction(stepper, NULL, EulerRHSFunction, this);
    TSSetFromOptions(stepper);
    return;
  } else {

    TSSetType(stepper, TSEIMEX);
    TSSetRHSFunction(stepper, NULL, EulerRHSFunction, this);

    DMCreateGlobalVector(qdomain, &Residual);
    DMCreateMatrix(qdomain, &Jacobian);

    TSSetIFunction(stepper, Residual, LHSIFunction2, this);
    TSSetIJacobian(stepper, Jacobian, Jacobian, LHSIJacobian2, this);

    // Allow for a couple of failed iterations before giving up
    PetscCallVoid(TSSetMaxSNESFailures(stepper, 5));
    TSSetFromOptions(stepper);
  }
}

Vischydro::~Vischydro() {
  if (not use_ideal_step_only) {
    VecDestroy(&Residual);
    MatDestroy(&Jacobian);
  }
  TSDestroy(&stepper);
  VecDestroy(&qsolution);
  DMDestroy(&qdomain);
  VecDestroy(&local_solution);
  VecDestroy(&solution);
  DMDestroy(&domain);
}
} // namespace DFHydro
