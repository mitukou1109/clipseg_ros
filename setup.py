from setuptools import find_packages, setup

package_name = "clipseg_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/segmentation.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mitsuhiro Sakamoto",
    maintainer_email="mitukou1109@gmail.com",
    description="ROS 2 wrapper for CLIPSeg",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "segmentation_node = clipseg_ros.segmentation:main",
        ],
    },
)
