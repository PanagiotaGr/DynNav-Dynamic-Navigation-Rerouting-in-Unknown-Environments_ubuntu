#ifndef DYNNAV_NAV2_CPP__DYNNAV_GLOBAL_PLANNER_HPP_
#define DYNNAV_NAV2_CPP__DYNNAV_GLOBAL_PLANNER_HPP_

#include <functional>
#include <memory>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "dynnav_nav2_cpp/grid_search.hpp"
#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"

namespace dynnav_nav2_cpp
{

class DynNavGlobalPlanner : public nav2_core::GlobalPlanner
{
public:
  DynNavGlobalPlanner() = default;
  ~DynNavGlobalPlanner() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal,
    std::function<bool()> cancel_checker) override;

private:
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::string name_;
  std::string global_frame_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Logger logger_{rclcpp::get_logger("dynnav_nav2_cpp")};
  GridSearchConfig search_config_;
};

}  // namespace dynnav_nav2_cpp

#endif  // DYNNAV_NAV2_CPP__DYNNAV_GLOBAL_PLANNER_HPP_
