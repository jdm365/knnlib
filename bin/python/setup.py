from setuptools import setup, find_packages


setup(
    name="knnlib",
    version="0.1.0",
    author="Jake Mehlman",
    description="A library for k-nearest neighbors",
    license="MIT",
    packages=find_packages(),
    package_data={"": ["*.so"]},
    include_package_data=True,
    )
