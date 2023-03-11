from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "my_script=my_package.script:main",
        ],
    },
)
