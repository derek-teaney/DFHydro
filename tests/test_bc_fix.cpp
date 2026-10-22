#include <DFHydro/Vischydro.hpp>
#include "Vischydro_impl.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <petscerror.h>

using namespace DFHydro;

// Returns the number of ghost cells with incorrect values on this rank.
int CheckPeriodicGhosts(VischydroNode **asol, const DMDALocalInfo &info) {
  int nx = info.mx, ny = info.my;
  int fail_count = 0;

  for (int j = info.gys; j < info.gys + info.gym; j++) {
    for (int i = info.gxs; i < info.gxs + info.gxm; i++) {
      bool owned = (i >= info.xs && i < info.xs + info.xm &&
                    j >= info.ys && j < info.ys + info.ym);
      if (owned) continue;

      // Expected value under periodic wrap: same formula as the fill loop below
      int wi = ((i % nx) + nx) % nx;
      int wj = ((j % ny) + ny) % ny;
      double expected = (double)(wj * nx + wi);
      double actual = asol[j][i].e;

      if (std::abs(actual - expected) > 1e-10) {
        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        PetscSynchronizedPrintf(PETSC_COMM_WORLD,
            "FAIL rank=%d ghost (%d,%d): expected %.1f got %.1f\n",
            rank, i, j, expected, actual);
        fail_count++;
      }
    }
  }
  return fail_count;
}

void Run(int &total_fails) {
  nlohmann::json input = nlohmann::json::parse(R"({
        "nx": 8,
        "ny": 6,
        "xmin": -20.0,
        "xmax": 20.0,
        "ymin": -15.0,
        "ymax": 15.0
  })");
  nlohmann::json config = R"({"is_periodic": true})"_json;

  ViscousQGP eos;
  eos.set_eta_by_s(4.0 / (4.0 * M_PI));

  int nx = input.at("nx").get<int>();
  int ny = input.at("ny").get<int>();
  double xmin = input.at("xmin").get<double>();
  double xmax = input.at("xmax").get<double>();
  double ymin = input.at("ymin").get<double>();
  double ymax = input.at("ymax").get<double>();
  Vischydro vischydro(nx, xmin, xmax, ny, ymin, ymax, &eos, config);

  DMDALocalInfo info;
  DMDAGetLocalInfo(vischydro.domain, &info);

  // Fill global solution: e[j][i] = j*nx + i (unique, analytically predictable)
  VischydroNode **asol;
  PetscCallVoid(DMDAVecGetArray(vischydro.domain, vischydro.solution, &asol));
  for (int i = info.xs; i < info.xs + info.xm; i++) {
    for (int j = info.ys; j < info.ys + info.ym; j++) {
      asol[j][i].e = (double)(j * nx + i);
    }
  }
  PetscCallVoid(DMDAVecRestoreArray(vischydro.domain, vischydro.solution, &asol));

  // Transfer global -> local (ghost cells filled periodically by PETSc DMDA)
  PetscCallVoid(DMGlobalToLocal(vischydro.domain, vischydro.solution,
                                INSERT_VALUES, vischydro.local_solution));
  PetscCallVoid(DMDAVecGetArray(vischydro.domain, vischydro.local_solution, &asol));

  // set_boundary_conditions returns immediately for periodic grids
  set_boundary_conditions(asol, info, vischydro.has_periodic_bc());

  int rank, size;
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // Write per-rank file for manual inspection
  std::string rank_filename = "rank_" + std::to_string(rank) + "_testbcfix.txt";
  std::ofstream ofs(rank_filename);
  std::stringstream ss;
  ss << "Rank " << rank << " has grid info:\n";
  ss << "  xs: " << info.xs << " ys: " << info.ys
     << " xm: " << info.xm << " ym: " << info.ym << "\n";
  ss << "  gxs: " << info.gxs << " gys: " << info.gys
     << " gxm: " << info.gxm << " gym: " << info.gym << "\n";
  PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%s", ss.str().c_str());
  for (PetscInt j = info.gys; j < info.gys + info.gym; j++) {
    ofs << j << " ";
    for (PetscInt i = info.gxs; i < info.gxs + info.gxm; i++) {
      ofs << asol[j][i].e << " ";
    }
    ofs << "\n";
  }

  int local_fails = CheckPeriodicGhosts(asol, info);
  PetscCallVoid(DMDAVecRestoreArray(vischydro.domain, vischydro.local_solution, &asol));

  PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

  MPI_Allreduce(&local_fails, &total_fails, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
}

int main(int argc, char **argv) {
  PetscInitialize(&argc, &argv, NULL, NULL);

  int total_fails = 0;
  Run(total_fails);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    if (total_fails == 0) {
      PetscPrintf(PETSC_COMM_WORLD, "PASS: all periodic ghost cells correct\n");
    } else {
      PetscPrintf(PETSC_COMM_WORLD, "FAIL: %d ghost cells had wrong values\n", total_fails);
    }
  }

  PetscFinalize();
  return (total_fails > 0) ? 1 : 0;
}
