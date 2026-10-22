#include <DFHydro/Vischydro.hpp>
#include <fstream>
#include <iostream>
#include <petscviewerhdf5.h>

using namespace DFHydro;

// This is a class to write out the grid to a file. It is used to write out the
// grid to an HDF5 file run_name_grid.h5 which contains the grid and an ascii
// file which contains the time and step number. The ascii file is
// run_name_grid_t.txt.  The information is written out every print_frequency
// timesteps.
class GridMonitorContext {
public:
  Vischydro *run;

  // Prints out the solution every print_frequency timesteps
  int print_frequency;

  // HDF5 file for the grid
  PetscViewer H5viewer;

  // Ascii file for the simplicity and small data
  std::ofstream ascii_file; 

  // A vector to hold the Landau frame variables
  Vec LFsolution;


  GridMonitorContext(nlohmann::json &input, Vischydro *run_in) : run(run_in) {
    print_frequency =
        input.at("VischydroMain").at("print_frequency").get<int>();

    // Create the HDF5 viewer  run_name + "_grid.h5"
    std::string run_name =
        input.at("VischydroMain").at("run_name").get<std::string>();
    std::string iofilename = run_name + "_grid.h5";

    PetscViewerHDF5Open(PETSC_COMM_WORLD, iofilename.c_str(), FILE_MODE_WRITE,
                        &H5viewer);
    PetscViewerSetFromOptions(H5viewer);
    PetscObjectSetName((PetscObject)run->coordinates, "coordinates");
    VecView(run->coordinates, H5viewer);
    PetscViewerHDF5PushTimestepping(H5viewer);

    // Create an ascii file
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      ascii_file.open(run_name + "_grid_t.txt");
    }

    DMCreateGlobalVector(run->domain, &LFsolution);
  }
  ~GridMonitorContext() {VecDestroy(&LFsolution); PetscViewerDestroy(&H5viewer); }
};

PetscErrorCode VischydroGridDFtoLDF(Vec solution, Vec LFsolution, Vischydro *run)
{
  // Get the solution on the charge grid
  VischydroNode **asol;
  PetscCall(DMDAVecGetArrayRead(run->domain, solution, &asol));

  // Get the array for the Landau frame variables
  VischydroNode **lfsol;
  PetscCall(DMDAVecGetArray(run->domain, LFsolution, &lfsol));

  // Get the corners of the domain
  PetscInt ixs, iys, ixm, iym;
  DMDAGetCorners(run->domain, &ixs, &iys, NULL, &ixm, &iym, NULL);

  for (PetscInt j = iys; j < iys + iym; j++) {
    for (PetscInt i = ixs; i < ixs + ixm; i++) {
      VischydroNode &node = asol[j][i];
      VischydroNode &lfnode = lfsol[j][i];
      vhnode_DFtoLF(*(run->eos), node, lfnode);
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(run->domain, solution, &asol));
  PetscCall(DMDAVecRestoreArray(run->domain, LFsolution, &lfsol));
  return 0;
}

// This is a monitor function that is called at each timestep by the TS Object
// It is used to write out the solution and works with GridMonitorContext to
// write out the grid.
PetscErrorCode VischydroGridMonitor(TS ts, PetscInt step, PetscReal time, Vec u,
                                    void *mctx) {

  auto monitor = (GridMonitorContext *)mctx;

  if (step % monitor->print_frequency == 0) {
    PetscPrintf(PETSC_COMM_WORLD, "Time, Step: %f %d \n", time, step);
    auto &solution = monitor->run->solution;
    auto &lfsolution = monitor->LFsolution;
    VischydroGridDFtoLDF(solution, lfsolution, monitor->run);

    PetscObjectSetName((PetscObject)solution, "solution");
    VecView(solution, monitor->H5viewer);

    PetscObjectSetName((PetscObject)lfsolution, "lfsolution");
    VecView(lfsolution, monitor->H5viewer);

    PetscViewerHDF5IncrementTimestep(monitor->H5viewer);

    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      monitor->ascii_file << time << " " << step << " "
                          << step / monitor->print_frequency << std::endl;
    }
  }

  return 0;
}

PetscErrorCode RunCode() {

  // Open the input file and parse the inputs into Json::Value
  char filename[PETSC_MAX_PATH_LEN] = "input.json";
  PetscOptionsGetString(NULL, NULL, "-input", filename, sizeof(filename), NULL);
  nlohmann::json input;
  std::ifstream ifs(filename);
  if (ifs) {
    ifs >> input;
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Unable to open input file %s. Aborting...\n",
                filename);
    return 0;
  }
  double etabys_in =
      input.at("VischydroMain").value("eta_by_s", 1. / (4. * M_PI));
  double zetabys_in = input.at("VischydroMain").value("zeta_by_s", 0.);

  // Create the EOS
  std::unique_ptr<EOS> eos =
      std::make_unique<ViscousQGP>(3., 0, etabys_in, zetabys_in);

  // Initialize the hydro
  int nx = input.at("VischydroMain").at("nx").get<int>();
  int ny = input.at("VischydroMain").at("ny").get<int>();
  double xmin = input.at("VischydroMain").at("xmin").get<double>();
  double xmax = input.at("VischydroMain").at("xmax").get<double>();
  double ymin = input.at("VischydroMain").at("ymin").get<double>();
  double ymax = input.at("VischydroMain").at("ymax").get<double>();

  nlohmann::json config =
      input.value("Vischydro", nlohmann::json::object());

  std::unique_ptr<Vischydro> vischydro =
      std::make_unique<Vischydro>(nx, xmin, xmax, ny, ymin, ymax, eos.get(), config);

  std::string ic_filename = input.at("VischydroMain")
                                .at("initial_conditions_filename")
                                .get<std::string>();
  
  std::string type = input.at("VischydroMain").value("initial_field_type", "primitives");
  vischydro->load_initial_conditions(ic_filename, type);

  // File names take the form run_name_initial.h5 and run_name_final.h5
  std::string run_name =
      input.at("VischydroMain").at("run_name").get<std::string>();

  vischydro->save(run_name + "_initial.h5");

  // Add a monitor to the stepper to print out the grid
  GridMonitorContext gctx(input, vischydro.get());
  TSMonitorSet(vischydro->stepper, VischydroGridMonitor, &gctx, NULL);

  double t_start = input.at("VischydroMain").at("t_start").get<double>();
  double t_end = input.at("VischydroMain").at("t_end").get<double>();
  double dt_max = input.at("VischydroMain").at("dt_max").get<double>();
  double dt = std::min(vischydro->get_default_time_step(), dt_max);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "initial_time: " << t_start << std::endl;
    std::cout << "dt: " << dt << std::endl;
    std::cout << "final_time: " << t_end << std::endl;
  }

  // Check the -help option to print out the input file structure
  PetscBool help = PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-help", &help, NULL);
  if (help) {
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      PetscPrintf(PETSC_COMM_WORLD, "This is the VischydroMain executable.\n");
      std::cout << std::setw(4) << input << std::endl;
    }
    return 0;
  }

  vischydro->solve(t_start, t_end, dt);

  // Save the final state
  if (rank == 0) {
    double t;
    TSGetTime(vischydro->stepper, &t);
    std::cout << "Final output time: " << t << std::endl;
  }
  vischydro->save(run_name + "_final.h5");
  return 0;
}

int main(int argc, char **argv) {
  PetscInitialize(&argc, &argv, NULL, NULL);
  RunCode();
  PetscFinalize();
}
