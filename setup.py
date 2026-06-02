import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="hygende",
    version="1.0.0",
    description="A package for hypothesis generation and inference hypotheses for depression detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "hygende_generation=hygende_cmd.generation:main",
            "hygende_inference=hygende_cmd.inference:main",
        ],
    },
)
