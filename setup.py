from setuptools import setup

setup(
    name="plot-hh4b-analysis",
    version="0.1",
    description="Code to plot the R22 HH4b analysis",
    url="https://github.com/ruelasvh/plot-hh4b-analysis.git",
    author="Victor Ruelas",
    author_email="victor.hugo.ruelas.rivera@cern.ch",
    license="MIT",
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "awkward>=2.0.4",
        "matplotlib>=3.6.2",
        "mplhep>=0.3.26",
        "numpy>=1.24.0",
        "scipy>=1.9.3",
        "uproot>=5.0.2",
        "vector>=0.11.0",
        "pyAMI-core==5.1.2",
        "pyAMI_atlas==5.1.0.1",
        "coloredlogs>=15.0.1",
    ],
    entry_points={
        "console_scripts": [
            "hh4b_non_res_res_make_hists = bin.hh4b_non_res_res_make_hists:main",
        ]
    },
    zip_safe=False,
)
