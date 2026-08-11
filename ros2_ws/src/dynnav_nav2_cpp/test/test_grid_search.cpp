#include <cstddef>
#include <cstdint>
#include <vector>

#include "dynnav_nav2_cpp/grid_search.hpp"
#include "gtest/gtest.h"

namespace dynnav_nav2_cpp
{
namespace
{

TEST(GridSearch, FindsObstacleAvoidingPath)
{
  constexpr std::size_t width = 5;
  constexpr std::size_t height = 5;
  std::vector<std::uint8_t> costs(width * height, 0U);
  for (std::size_t y = 0; y < height - 1; ++y) {
    costs[y * width + 2] = 254U;
  }

  GridSearchConfig config;
  config.risk_weight = 0.0;
  config.irreversibility_weight = 0.0;
  const auto result = planGridPath(width, height, costs, 0, 4, config);

  ASSERT_TRUE(result.success);
  ASSERT_GT(result.path.size(), 5U);
  for (const auto index : result.path) {
    EXPECT_LT(costs[index], config.lethal_cost_threshold);
  }
}

TEST(GridSearch, RiskWeightSelectsLongerLowRiskRoute)
{
  constexpr std::size_t width = 5;
  constexpr std::size_t height = 3;
  std::vector<std::uint8_t> costs(width * height, 0U);
  costs[1 * width + 1] = 200U;
  costs[1 * width + 2] = 200U;
  costs[1 * width + 3] = 200U;

  GridSearchConfig shortest;
  shortest.risk_weight = 0.0;
  shortest.irreversibility_weight = 0.0;
  const auto shortest_result = planGridPath(width, height, costs, 5, 9, shortest);

  GridSearchConfig risk_aware = shortest;
  risk_aware.risk_weight = 20.0;
  const auto risk_result = planGridPath(width, height, costs, 5, 9, risk_aware);

  ASSERT_TRUE(shortest_result.success);
  ASSERT_TRUE(risk_result.success);
  EXPECT_GT(risk_result.path.size(), shortest_result.path.size());
  for (const auto index : risk_result.path) {
    EXPECT_NE(costs[index], 200U);
  }
}

TEST(GridSearch, OpenCellIsMoreRecoverableThanCorridorCell)
{
  constexpr std::size_t width = 5;
  constexpr std::size_t height = 5;
  GridSearchConfig config;
  std::vector<std::uint8_t> open_costs(width * height, 0U);
  std::vector<std::uint8_t> corridor_costs(width * height, 254U);
  for (std::size_t x = 0; x < width; ++x) {
    corridor_costs[2 * width + x] = 0U;
  }

  const auto open_score = localIrreversibility(12, width, height, open_costs, config);
  const auto corridor_score = localIrreversibility(12, width, height, corridor_costs, config);
  EXPECT_LT(open_score, corridor_score);
}

TEST(GridSearch, RespectsUnknownPolicyAndCancellation)
{
  constexpr std::size_t width = 3;
  constexpr std::size_t height = 1;
  const std::vector<std::uint8_t> costs{0U, 255U, 0U};

  GridSearchConfig blocked;
  blocked.allow_unknown = false;
  EXPECT_FALSE(planGridPath(width, height, costs, 0, 2, blocked).success);

  GridSearchConfig allowed = blocked;
  allowed.allow_unknown = true;
  EXPECT_TRUE(planGridPath(width, height, costs, 0, 2, allowed).success);

  const auto cancelled = planGridPath(
    width, height, costs, 0, 2, allowed, []() {return true;});
  EXPECT_TRUE(cancelled.cancelled);
  EXPECT_FALSE(cancelled.success);
}

}  // namespace
}  // namespace dynnav_nav2_cpp
