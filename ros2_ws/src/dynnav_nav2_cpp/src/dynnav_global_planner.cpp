#include "dynnav_nav2_cpp/dynnav_global_planner.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nav2_core/planner_exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace dynnav_nav2_cpp
{

void DynNavGlobalPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  name_ = name;
  tf_ = std::move(tf);
  costmap_ros_ = std::move(costmap_ros);
  costmap_ = costmap_ros_->getCostmap();
  global_frame_ = costmap_ros_->getGlobalFrameID();

  const auto node = parent.lock();
  if (!node) {
    throw nav2_core::PlannerException("DynNav planner parent lifecycle node expired");
  }
  clock_ = node->get_clock();
  logger_ = node->get_logger();

  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".allow_unknown", rclcpp::ParameterValue(true));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".lethal_cost_threshold", rclcpp::ParameterValue(253));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".neutral_cost", rclcpp::ParameterValue(1.0));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".risk_weight", rclcpp::ParameterValue(4.0));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".irreversibility_weight", rclcpp::ParameterValue(4.0));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".unknown_risk", rclcpp::ParameterValue(0.5));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".max_iterations", rclcpp::ParameterValue(0));

  int lethal_cost_threshold = 253;
  int max_iterations = 0;
  node->get_parameter(name_ + ".allow_unknown", search_config_.allow_unknown);
  node->get_parameter(name_ + ".lethal_cost_threshold", lethal_cost_threshold);
  node->get_parameter(name_ + ".neutral_cost", search_config_.neutral_cost);
  node->get_parameter(name_ + ".risk_weight", search_config_.risk_weight);
  node->get_parameter(name_ + ".irreversibility_weight", search_config_.irreversibility_weight);
  node->get_parameter(name_ + ".unknown_risk", search_config_.unknown_risk);
  node->get_parameter(name_ + ".max_iterations", max_iterations);

  if (lethal_cost_threshold < 1 || lethal_cost_threshold > 254) {
    throw nav2_core::PlannerException("lethal_cost_threshold must be in [1, 254]");
  }
  if (max_iterations < 0) {
    throw nav2_core::PlannerException("max_iterations must be non-negative");
  }
  search_config_.lethal_cost_threshold = static_cast<std::uint8_t>(lethal_cost_threshold);
  search_config_.max_iterations = static_cast<std::size_t>(max_iterations);
  validateGridSearchConfig(search_config_);

  RCLCPP_INFO(
    logger_,
    "Configured %s: risk_weight=%.3f irreversibility_weight=%.3f allow_unknown=%s",
    name_.c_str(), search_config_.risk_weight, search_config_.irreversibility_weight,
    search_config_.allow_unknown ? "true" : "false");
}

void DynNavGlobalPlanner::cleanup()
{
  RCLCPP_INFO(logger_, "Cleaning up DynNav planner %s", name_.c_str());
  costmap_ = nullptr;
  costmap_ros_.reset();
  tf_.reset();
  clock_.reset();
}

void DynNavGlobalPlanner::activate()
{
  RCLCPP_INFO(logger_, "Activating DynNav planner %s", name_.c_str());
}

void DynNavGlobalPlanner::deactivate()
{
  RCLCPP_INFO(logger_, "Deactivating DynNav planner %s", name_.c_str());
}

nav_msgs::msg::Path DynNavGlobalPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal,
  std::function<bool()> cancel_checker)
{
  if (costmap_ == nullptr || clock_ == nullptr) {
    throw nav2_core::PlannerException("DynNav planner has not been configured");
  }
  if (start.header.frame_id != global_frame_ || goal.header.frame_id != global_frame_) {
    throw nav2_core::PlannerTFError(
            "Start and goal must be expressed in the global costmap frame " + global_frame_);
  }

  unsigned int start_x = 0;
  unsigned int start_y = 0;
  unsigned int goal_x = 0;
  unsigned int goal_y = 0;
  if (!costmap_->worldToMap(start.pose.position.x, start.pose.position.y, start_x, start_y)) {
    throw nav2_core::StartOutsideMapBounds("Start pose is outside the global costmap");
  }
  if (!costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, goal_x, goal_y)) {
    throw nav2_core::GoalOutsideMapBounds("Goal pose is outside the global costmap");
  }

  const auto width = static_cast<std::size_t>(costmap_->getSizeInCellsX());
  const auto height = static_cast<std::size_t>(costmap_->getSizeInCellsY());
  const auto start_index = static_cast<std::size_t>(costmap_->getIndex(start_x, start_y));
  const auto goal_index = static_cast<std::size_t>(costmap_->getIndex(goal_x, goal_y));

  std::vector<std::uint8_t> costs;
  {
    std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap_->getMutex()));
    const auto * char_map = costmap_->getCharMap();
    costs.assign(char_map, char_map + width * height);
  }
  costs[start_index] = 0U;
  if (!isTraversable(costs[goal_index], search_config_)) {
    throw nav2_core::GoalOccupied("Goal cell is not traversable");
  }

  const auto result = planGridPath(
    width, height, costs, start_index, goal_index, search_config_, cancel_checker);
  if (result.cancelled) {
    throw nav2_core::PlannerCancelled("DynNav planning request was cancelled");
  }
  if (!result.success) {
    throw nav2_core::NoValidPathCouldBeFound("DynNav could not find a traversable path");
  }

  nav_msgs::msg::Path path;
  path.header.stamp = clock_->now();
  path.header.frame_id = global_frame_;
  path.poses.reserve(result.path.size());

  for (const auto index : result.path) {
    const auto map_x = static_cast<unsigned int>(index % width);
    const auto map_y = static_cast<unsigned int>(index / width);
    double world_x = 0.0;
    double world_y = 0.0;
    costmap_->mapToWorld(map_x, map_y, world_x, world_y);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = world_x;
    pose.pose.position.y = world_y;
    pose.pose.orientation.w = 1.0;
    path.poses.push_back(pose);
  }

  path.poses.front().pose.position = start.pose.position;
  path.poses.back().pose = goal.pose;
  for (std::size_t index = 0; index + 1 < path.poses.size(); ++index) {
    const auto & current = path.poses[index].pose.position;
    const auto & next = path.poses[index + 1].pose.position;
    tf2::Quaternion orientation;
    orientation.setRPY(0.0, 0.0, std::atan2(next.y - current.y, next.x - current.x));
    path.poses[index].pose.orientation = tf2::toMsg(orientation);
  }

  RCLCPP_DEBUG(
    logger_, "DynNav path: %zu poses, %zu expanded nodes, objective %.3f",
    path.poses.size(), result.expanded_nodes, result.total_cost);
  return path;
}

}  // namespace dynnav_nav2_cpp

PLUGINLIB_EXPORT_CLASS(dynnav_nav2_cpp::DynNavGlobalPlanner, nav2_core::GlobalPlanner)
