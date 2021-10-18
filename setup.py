import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="explainable_ai_image_measures",
    version="1.0.1",
    description="Compute IAUC, DAUC, IROF scores to measure quality of image attributions",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Meier-Johannes/ExplainableAIImageMeasures",
    author="Johannes Meier",
    author_email="johannes-michael.meier@student.uni-tuebingen.de",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["explainable_ai_image_measures"],
    include_package_data=True,
    install_requires=["numpy", "torch", "scikit-image", "scikit-learn"],
)
