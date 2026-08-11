from setuptools import setup

package_name = "dynnav_nav2"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/dynnav_planner_bridge.launch.py"]),
        ("share/" + package_name + "/config", ["config/dynnav_planner_bridge.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Panagiota Grosdouli",
    maintainer_email="75089541+panagiotagrosdouli@users.noreply.github.com",
    description="Diagnostic ROS 2 bridge for the Python DynNav research planners.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dynnav_planner_bridge = dynnav_nav2.dynnav_planner_bridge:main",
        ],
    },
)
