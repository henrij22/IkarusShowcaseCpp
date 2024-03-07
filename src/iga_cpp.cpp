// SPDX-FileCopyrightText: 2024 The Ikarus Developers jakob@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifdef HAVE_CONFIG_H
  #include "config.h"
#endif

#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/iga/io/ibra/ibrareader.hh>
#include <dune/iga/io/igadatacollector.hh>
#include <dune/iga/nurbsbasis.hh>
#include <dune/iga/nurbsgrid.hh>
#include <dune/iga/utils/igahelpers.hh>
#include <dune/vtk/vtkwriter.hh>
#include <dune/vtk/writers/unstructuredgridwriter.hh>

#include <ikarus/assembler/simpleassemblers.hh>
#include <ikarus/controlroutines/loadcontrol.hh>
#include <ikarus/finiteelements/mechanics/kirchhoffloveshell.hh>
#include <ikarus/solver/nonlinearsolver/newtonraphson.hh>
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/dirichletvalues.hh>
#include <ikarus/utils/init.hh>

int main(int argc, char** argv) {
  Ikarus::init(argc, argv);

  /// Defs
  const double nu   = 0.0;
  const double Emod = 1000;
  const double thk  = 0.1;

  const bool trim = true;

  using Grid     = Dune::IGA::NURBSGrid<2, 3>;
  using GridView = Grid::LeafGridView;

  const auto grid = Dune::IGA::IbraReader<2, 3>::read("input/plate_holes.ibra", trim);
  grid->globalRefine(6);
  const GridView gridView = grid->leafGridView();

  using namespace Dune::Functions::BasisFactory;
  auto basis = Ikarus::makeBasis(gridView, power<3>(nurbs(), FlatInterleaved()));

  Ikarus::DirichletValues dirichletValues(basis.flat());

  dirichletValues.fixDOFs([](auto& basis_, auto& dirichletFlags) {
    Dune::Functions::forEachUntrimmedBoundaryDOF(Dune::Functions::subspaceBasis(basis_, 2),
                                                 [&](auto&& localIndex, auto&& localView, auto&& intersection) {
                                                   dirichletFlags[localView.index(localIndex)] = true;
                                                 });
    auto fixEverything = [&](auto&& subBasis_) {
      auto localView       = subBasis_.localView();
      auto seDOFs          = subEntityDOFs(subBasis_);
      const auto& gridView = subBasis_.gridView();
      for (auto&& element : elements(gridView)) {
        localView.bind(element);
        for (const auto& intersection : intersections(gridView, element))
          for (auto localIndex : seDOFs.bind(localView, intersection))
            dirichletFlags[localView.index(localIndex)] = true;
      }
    };
    fixEverything(Dune::Functions::subspaceBasis(basis_, 0));
    fixEverything(Dune::Functions::subspaceBasis(basis_, 1));
  });

  auto volumeLoad = [thk]([[maybe_unused]] auto& globalCoord, auto& lamb) {
    return Eigen::Vector3d{0, 0, 2 * Dune::power(thk, 3) * lamb};
  };

  using KLShell = Ikarus::KirchhoffLoveShell<decltype(basis)>;
  std::vector<KLShell> fes;

  for (auto& element : Dune::elements(gridView))
    fes.emplace_back(basis, element, Emod, nu, thk, volumeLoad);

  /// Create a sparse assembler
  auto sparseAssembler = Ikarus::SparseFlatAssembler(fes, dirichletValues);

  /// Define "elastoStatics" affordances and create functions for stiffness matrix and residual calculations
  auto req = Ikarus::FErequirements().addAffordance(Ikarus::AffordanceCollections::elastoStatics);

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

  double lambda     = 0.0;
  Eigen::VectorXd d = Eigen::VectorXd::Zero(basis.flat().size());

  auto nonLinOp =
      Ikarus::NonLinearOperator(Ikarus::functions(residualFunction, KFunction), Ikarus::parameter(d, lambda));

  Ikarus::LinearSolver linSolver{Ikarus::SolverTypeTag::si_ConjugateGradient};
  auto solver = Ikarus::makeNewtonRaphson(nonLinOp, std::move(linSolver));
  solver->setup({1e-8, 300});

  auto lc                                           = Ikarus::LoadControl(solver, 1, {0, 1});
  const auto [success, solverInfo, totalIterations] = lc.run();

  std::cout << std::boolalpha << success << " " << totalIterations << std::endl;
  std::cout << std::ranges::max(d) << std::endl;
  auto dispGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, 3>>(basis.flat(), d);

  Dune::Vtk::DiscontinuousIgaDataCollector dataCollector(gridView, 0);
  Dune::Vtk::UnstructuredGridWriter vtkWriter(dataCollector, Dune::Vtk::FormatTypes::ASCII);

  vtkWriter.addPointData(dispGlobalFunc, Dune::VTK::FieldInfo("displacement", Dune::VTK::FieldInfo::Type::vector, 3));

  vtkWriter.write("result");

  return 0;
}
