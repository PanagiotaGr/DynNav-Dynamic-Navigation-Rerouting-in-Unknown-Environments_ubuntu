#include <memory>

#include "gtest/gtest.h"
#include "nav2_core/global_planner.hpp"
#include "pluginlib/class_loader.hpp"

TEST(DynNavPlugin, IsDiscoverableAndInstantiable)
{
  pluginlib::ClassLoader<nav2_core::GlobalPlanner> loader(
    "nav2_core", "nav2_core::GlobalPlanner");
  const auto planner = loader.createSharedInstance("dynnav_nav2_cpp::DynNavGlobalPlanner");
  ASSERT_NE(planner, nullptr);
}
