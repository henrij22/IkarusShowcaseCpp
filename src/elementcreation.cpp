// SPDX-FileCopyrightText: 2024 Henrik Jakob jakob@ibb.uni-stuttgart.de
// SPDX-License-Identifier: MIT

#ifdef HAVE_CONFIG_H
  #include "config.h"
#endif

#include <dune/grid/yaspgrid.hh>

#include <ikarus/finiteelements/fefactory.hh>
#include <ikarus/finiteelements/mechanics/linearelastic.hh>
#include <ikarus/finiteelements/mechanics/enhancedassumedstrains.hh>
#include <ikarus/finiteelements/mechanics/loads.hh>
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/init.hh>

int main(int argc, char** argv) {
  Ikarus::init(argc, argv);

  using Grid = Dune::YaspGrid<2>;

  Dune::FieldVector<double, 2> boundingBox{1, 1};
  std::array elementsPerDirection{10, 10};
  auto grid     = std::make_shared<Grid>(boundingBox, elementsPerDirection);
  auto gridView = grid->leafGridView();

  using namespace Dune::Functions::BasisFactory;
  auto basis = Ikarus::makeBasis(gridView, power<2>(lagrange<1>()));

  auto vL = [](auto& globalCoord, auto& loadFactor) {
    return Eigen::Vector2d{loadFactor * globalCoord[0] * 2, 2 * loadFactor * globalCoord[1]};
  };

  auto skills = Ikarus::skills(Ikarus::linearElastic({.emodul=1000, .nu=0.2}), Ikarus::volumeLoad<2>(vL), Ikarus::eas(4));

  auto fe = Ikarus::makeFE(basis, skills);
}