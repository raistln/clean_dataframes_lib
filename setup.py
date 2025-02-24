from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clean_df_lib",
    version="0.1.0",
    author="Samuel Martín",
    author_email="samumarfon@gmail.com",
    description="A package for data cleaning and analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raistln/clean_dataframes_lib",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "chardet",
        "rapidfuzz",
        "nltk"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    keywords="data cleaning, data analysis, pandas, numpy",
    package_data={
        "": ["README.md", "LICENSE"],
    },
    python_requires=">=3.7",
)