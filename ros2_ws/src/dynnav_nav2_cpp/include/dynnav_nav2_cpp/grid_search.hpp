#ifndef DYNNAV_NAV2_CPP__GRID_SEARCH_HPP_
#define DYNNAV_NAV2_CPP__GRID_SEARCH_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace dynnav_nav2_cpp
{

struct GridSearchConfig
{
  bool allow_unknown{true};
  std::uint8_t lethal_cost_threshold{253};
  std::uint8_t unknown_cost{255};
  double neutral_cost{1.0};
  double risk_weight{4.0};
  double irreversibility_weight{4.0};
  double unknown_risk{0.5};
  std::size_t max_iterations{0};
};

struct GridSearchResult
{
  bool success{false};
  bool cancelled{false};
  std::vector<std::size_t> path;
  std::size_t expanded_nodes{0};
  double total_cost{0.0};
};

void validateGridSearchConfig(const GridSearchConfig & config);

bool isTraversable(std::uint8_t cost, const GridSearchConfig & config);

double normalizedRisk(std::uint8_t cost, const GridSearchConfig & config);

double localIrreversibility(
  std::size_t index,
  std::size_t width,
  std::size_t height,
  const std::vector<std::uint8_t> & costs,
  const GridSearchConfig & config);

GridSearchResult planGridPath(
  std::size_t width,
  std::size_t height,
  const std::vector<std::uint8_t> & costs,
  std::size_t start_index,
  std::size_t goal_index,
  const GridSearchConfig & config,
  const std::function<bool()> & cancel_checker = []() {return false;});

}  // namespace dynnav_nav2_cpp

#endif  // DYNNAV_NAV2_CPP__GRID_SEARCH_HPP_
