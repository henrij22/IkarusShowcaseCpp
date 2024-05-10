// SPDX-FileCopyrightText: 2024 Henrik Jakob jakob@ibb.uni-stuttgart.de
// SPDX-License-Identifier: MIT

#ifdef HAVE_CONFIG_H
  #include "config.h"
#endif

#include "timer.h"

#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/iga/hierarchicpatch/patchgrid.hh>
#include <dune/iga/io/ibrareader.hh>
#include <dune/iga/io/vtk/igadatacollector.hh>
#include <dune/iga/nurbsbasis.hh>
#include <dune/iga/trimmer/defaulttrimmer/trimmer.hh>
#include <dune/iga/utils/igahelpers.hh>
#include <dune/vtk/vtkwriter.hh>
#include <dune/vtk/writers/unstructuredgridwriter.hh>

#include <ikarus/assembler/simpleassemblers.hh>
#include <ikarus/controlroutines/loadcontrol.hh>
#include <ikarus/finiteelements/fefactory.hh>
#include <ikarus/finiteelements/mechanics/kirchhoffloveshell.hh>
#include <ikarus/finiteelements/mechanics/loads.hh>
#include <ikarus/solver/linearsolver/linearsolver.hh>
#include <ikarus/solver/nonlinearsolver/newtonraphson.hh>
#include <ikarus/solver/nonlinearsolver/trustregion.hh>
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/dirichletvalues.hh>
#include <ikarus/utils/functionhelper.hh>
#include <ikarus/utils/init.hh>
#include <ikarus/utils/linearalgebrahelper.hh>
#include <ikarus/utils/nonlinearoperator.hh>
#include <ikarus/utils/observer/genericobserver.hh>
#include <ikarus/utils/observer/observermessages.hh>


using namespace Dune::IGANEW;

auto run_calculation(int degree, int refinement) {
  /// Defs
  const double nu   = 0.0;
  const double Emod = 1000;
  const double thk  = 0.1;
  const bool trim   = true;

  using PatchGrid   = PatchGrid<2, 3, DefaultTrim::PatchGridFamily>;
  using GridFactory = Dune::GridFactory<PatchGrid>;

  using GridView = PatchGrid::LeafGridView;

  auto igaGridFactory = GridFactory();
  igaGridFactory.insertJson("input/plate_holes.ibra", true, {refinement, refinement});
  igaGridFactory.insertTrimParameters(GridFactory::TrimParameterType{120});

  auto grid = igaGridFactory.createGrid();
  grid->degreeElevateOnAllLevels({degree -1, degree -1});

  const GridView gridView = grid->leafGridView();

  using namespace Dune::Functions::BasisFactory;
  auto basis = Ikarus::makeBasis(gridView, power<3>(nurbs(), FlatInterleaved()));

  Ikarus::DirichletValues dirichletValues(basis.flat());

  dirichletValues.fixDOFs([](auto& basis_, auto& dirichletFlags) {
    Dune::Functions::forEachUntrimmedBoundaryDOF(basis_, [&](auto&& localIndex, auto&& localView, auto&& intersection) {
      dirichletFlags[localView.index(localIndex)] = true;
    });
  });

  auto vL = [thk]([[maybe_unused]] auto& globalCoord, auto& lamb) {
    return Eigen::Vector3d{0, 0, 2 * Dune::power(thk, 3) * lamb};
  };

  auto sk = Ikarus::skills(Ikarus::kirchhoffLoveShell({.youngs_modulus = Emod, .nu = nu, .thickness = thk}),
                           Ikarus::volumeLoad<3>(vL));
 
  using KLShell = decltype(Ikarus::makeFE(basis, sk));
  std::vector<KLShell> fes;

  for (auto&& element : elements(gridView)) {
    fes.emplace_back(Ikarus::makeFE(basis, sk));
    fes.back().bind(element);
  }

  // Set parameters for integration rule 
  Preferences::getInstance().boundaryDivisions(8);

  /// Create a sparse assembler
  auto sparseAssembler = Ikarus::SparseFlatAssembler(fes, dirichletValues);

  /// Define "elastoStatics" affordances and create functions for stiffness matrix and residual calculations
  auto req        = Ikarus::FErequirements().addAffordance(Ikarus::AffordanceCollections::elastoStatics);
  auto lambdaLoad = 1.0;
  req.insertParameter(Ikarus::FEParameter::loadfactor, lambdaLoad);

  auto residualFunction = [&](auto&& disp_, auto&& lambdaLocal) -> auto& {
    req.insertGlobalSolution(Ikarus::FESolutions::displacement, disp_)
        .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal);
    return sparseAssembler.getVector(req);
  };

  auto KFunction = [&](auto&& disp_, auto&& lambdaLocal) -> auto& {
    req.insertGlobalSolution(Ikarus::FESolutions::displacement, disp_)
        .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal);
    return sparseAssembler.getMatrix(req);
  };

  auto energyFunction = [&](auto&& disp_, auto&& lambdaLocal) -> auto& {
    req.insertGlobalSolution(Ikarus::FESolutions::displacement, disp_)
        .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal);
    return sparseAssembler.getScalar(req);
  };

  Eigen::VectorXd d = Eigen::VectorXd::Zero(basis.flat().size());

  auto nonLinOp = Ikarus::NonLinearOperator(Ikarus::functions(energyFunction, residualFunction, KFunction),
                                            Ikarus::parameter(d, lambdaLoad));

  auto trustRegion   = Ikarus::makeTrustRegion(nonLinOp);
  auto settings      = Ikarus::TrustRegionSettings{};
  settings.maxIter   = 300;
  settings.grad_tol  = 1e-8;
  settings.verbosity = 0;
  trustRegion->setup(settings);

  auto informations = trustRegion->solve();

  auto dispGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, 3>>(basis.flat(), d);

  Dune::Vtk::DiscontinuousIgaDataCollector dataCollector(gridView, 0);
  Dune::Vtk::UnstructuredGridWriter vtkWriter(dataCollector, Dune::Vtk::FormatTypes::ASCII);

  vtkWriter.addPointData(dispGlobalFunc, Dune::VTK::FieldInfo("displacement", Dune::VTK::FieldInfo::Type::vector, 3));

  vtkWriter.write("result_r" + std::to_string(refinement) + "_d" + std::to_string(degree));


  return std::make_tuple(informations.iterations,
                         sparseAssembler.reducedSize());
}

int main(int argc, char* argv[]) {
  Ikarus::init(argc, argv);
  Timer<> timer{};

  // If we are in testing mode (e.g. through GH Action, we only run one iteration)
  bool testing = argc > 1 && std::strcmp(argv[1], "testing") == 0;

  auto degreeRange     = Dune::range(2, testing ? 3 : 3);
  auto refinementRange = Dune::range(3, testing ? 4 : 5);

  std::vector<std::tuple<int, int, int, int, Timer<>::Period>> results{};
  for (auto i : degreeRange) {
    for (auto j : refinementRange) {
      timer.startTimer("total");
      auto [iterations, dofs] = run_calculation(i, j);
      results.emplace_back(i, j, iterations, dofs, timer.stopTimer("total").count());
    }
  }

  // for (auto [degree, refinement, max_d, iterations, dofs, seconds] : results)
  //   std::cout << "Degree: " << degree << ", Ref: " << refinement << ", max_d: " << max_d
  //             << ", iterations: " << iterations << ", Dofs: " << dofs << ", Compute time: " << seconds << std::endl;
  return 0;
}
