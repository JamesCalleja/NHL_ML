from setuptools import setup, find_packages

setup(
    name="nhl",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.31.0",
        "pandas>=2.1.0",
        "nhlpy>=0.3.0",
    ],
    author="James Calleja",
    author_email="james.calleja@gmail.com",
    description="NHL Machine Learning Tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JamesCalleja/nhl-ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 