#include "dynnav_nav2_cpp/grid_search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dynnav_nav2_cpp
{
namespace
{

constexpr double kEpsilon = 1.0e-12;
constexpr std::size_t kNoParent = std::numeric_limits<std::size_t>::max();

struct QueueEntry
{
  double priority;
  double cost;
  std::size_t index;
  std::size_t order;
};

struct GreaterPriority
{
  bool operator()(const QueueEntry & left, const QueueEntry & right) const
  {
    if (std::abs(left.priority - right.priority) > kEpsilon) {
      return left.priority > right.priority;
    }
    return left.order > right.order;
  }
};

std::size_t manhattan(
  const std::size_t left,
  const std::size_t right,
  const std::size_t width)
{
  const auto left_x = left % width;
  const auto left_y = left / width;
  const auto right_x = right % width;
  const auto right_y = right / width;
  return
    (left_x > right_x ? left_x - right_x : right_x - left_x) +
    (left_y > right_y ? left_y - right_y : right_y - left_y);
}

std::vector<std::size_t> neighbours4(
  const std::size_t index,
  const std::size_t width,
  const std::size_t height)
{
  const auto x = index % width;
  const auto y = index / width;
  std::vector<std::size_t> neighbours;
  neighbours.reserve(4);
  if (x + 1 < width) {
    neighbours.push_back(index + 1);
  }
  if (x > 0) {
    neighbours.push_back(index - 1);
  }
  if (y + 1 < height) {
    neighbours.push_back(index + width);
  }
  if (y > 0) {
    neighbours.push_back(index - width);
  }
  return neighbours;
}

}  // namespace

void validateGridSearchConfig(const GridSearchConfig & config)
{
  if (config.lethal_cost_threshold == 0 ||
    config.lethal_cost_threshold >= config.unknown_cost)
  {
    throw std::invalid_argument("lethal_cost_threshold must be in [1, unknown_cost)");
  }
  if (!std::isfinite(config.neutral_cost) || config.neutral_cost <= 0.0) {
    throw std::invalid_argument("neutral_cost must be finite and positive");
  }
  if (!std::isfinite(config.risk_weight) || config.risk_weight < 0.0) {
    throw std::invalid_argument("risk_weight must be finite and non-negative");
  }
  if (!std::isfinite(config.irreversibility_weight) || config.irreversibility_weight < 0.0) {
    throw std::invalid_argument("irreversibility_weight must be finite and non-negative");
  }
  if (!std::isfinite(config.unknown_risk) ||
    config.unknown_risk < 0.0 || config.unknown_risk > 1.0)
  {
    throw std::invalid_argument("unknown_risk must be in [0, 1]");
  }
}

bool isTraversable(const std::uint8_t cost, const GridSearchConfig & config)
{
  if (cost == config.unknown_cost) {
    return config.allow_unknown;
  }
  return cost < config.lethal_cost_threshold;
}

double normalizedRisk(const std::uint8_t cost, const GridSearchConfig & config)
{
  if (cost == config.unknown_cost) {
    return config.unknown_risk;
  }
  const auto denominator = static_cast<double>(config.lethal_cost_threshold - 1U);
  return std::clamp(static_cast<double>(cost) / denominator, 0.0, 1.0);
}

double localIrreversibility(
  const std::size_t index,
  const std::size_t width,
  const std::size_t height,
  const std::vector<std::uint8_t> & costs,
  const GridSearchConfig & config)
{
  if (width == 0 || height == 0 || costs.size() != width * height || index >= costs.size()) {
    throw std::invalid_argument("invalid grid supplied to localIrreversibility");
  }

  std::size_t escape_options = 0;
  for (const auto neighbour : neighbours4(index, width, height)) {
    if (isTraversable(costs[neighbour], config)) {
      ++escape_options;
    }
  }

  const double escape_deficit = 1.0 / (1.0 + static_cast<double>(escape_options));
  double bottleneck_exposure = 0.0;
  if (escape_options <= 1) {
    bottleneck_exposure = 1.0;
  } else if (escape_options == 2) {
    bottleneck_exposure = 0.75;
  } else if (escape_options == 3) {
    bottleneck_exposure = 0.25;
  }
  return 0.5 * (escape_deficit + bottleneck_exposure);
}

GridSearchResult planGridPath(
  const std::size_t width,
  const std::size_t height,
  const std::vector<std::uint8_t> & costs,
  const std::size_t start_index,
  const std::size_t goal_index,
  const GridSearchConfig & config,
  const std::function<bool()> & cancel_checker)
{
  validateGridSearchConfig(config);
  if (width == 0 || height == 0 || costs.size() != width * height) {
    throw std::invalid_argument("grid dimensions do not match the cost array");
  }
  if (start_index >= costs.size() || goal_index >= costs.size()) {
    throw std::out_of_range("start or goal index is outside the grid");
  }

  GridSearchResult result;
  if (!isTraversable(costs[start_index], config) || !isTraversable(costs[goal_index], config)) {
    return result;
  }
  if (start_index == goal_index) {
    result.success = true;
    result.path = {start_index};
    return result;
  }

  const auto infinity = std::numeric_limits<double>::infinity();
  std::vector<double> cost_so_far(costs.size(), infinity);
  std::vector<std::size_t> parent(costs.size(), kNoParent);
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, GreaterPriority> frontier;

  std::size_t order = 0;
  cost_so_far[start_index] = 0.0;
  frontier.push({
    config.neutral_cost * static_cast<double>(manhattan(start_index, goal_index, width)),
    0.0, start_index, order});

  while (!frontier.empty()) {
    if (cancel_checker && cancel_checker()) {
      result.cancelled = true;
      return result;
    }

    const auto current = frontier.top();
    frontier.pop();
    if (current.cost > cost_so_far[current.index] + kEpsilon) {
      continue;
    }

    if (config.max_iterations > 0 && result.expanded_nodes >= config.max_iterations) {
      return result;
    }
    ++result.expanded_nodes;
    if (current.index == goal_index) {
      result.success = true;
      result.total_cost = cost_so_far[goal_index];
      for (auto cursor = goal_index; cursor != kNoParent; cursor = parent[cursor]) {
        result.path.push_back(cursor);
        if (cursor == start_index) {
          break;
        }
      }
      if (result.path.empty() || result.path.back() != start_index) {
        return GridSearchResult{};
      }
      std::reverse(result.path.begin(), result.path.end());
      return result;
    }

    for (const auto neighbour : neighbours4(current.index, width, height)) {
      if (!isTraversable(costs[neighbour], config)) {
        continue;
      }
      const double transition_cost =
        config.neutral_cost +
        config.risk_weight * normalizedRisk(costs[neighbour], config) +
        config.irreversibility_weight *
        localIrreversibility(neighbour, width, height, costs, config);
      const double candidate_cost = cost_so_far[current.index] + transition_cost;
      if (candidate_cost + kEpsilon < cost_so_far[neighbour]) {
        cost_so_far[neighbour] = candidate_cost;
        parent[neighbour] = current.index;
        ++order;
        const double heuristic = config.neutral_cost * static_cast<double>(
          manhattan(neighbour, goal_index, width));
        frontier.push({candidate_cost + heuristic, candidate_cost, neighbour, order});
      }
    }
  }

  return result;
}

}  // namespace dynnav_nav2_cpp
