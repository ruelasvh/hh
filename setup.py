from setuptools import setup

setup(
    name="hh",
    version="0.1",
    description="Code to plot the R21 and R22 HH4b analysis",
    url="https://github.com/ruelasvh/hh.git",
    author="Victor Ruelas",
    author_email="victor.hugo.ruelas.rivera@cern.ch",
    license="MIT",
    packages=[""],
    python_requires=">=3.7",
    install_requires=[
        "awkward>=2.2.4",
        "matplotlib>=3.6.2",
        "mplhep>=0.3.26",
        "numpy>=1.24.0",
        "scipy>=1.9.3",
        "uproot>=5.0.2",
        "vector>=0.11.0",
        "coloredlogs>=15.0.1",
        "h5py>=3.8.0",
        "pandas>=2.1.0",
        "tables>=3.9.2",
        "htcondor>=10.6.0",
    ],
    entry_points={
        "console_scripts": [
            "hh4b_non_res_res_make_hists = cli.make_hists_non_res_res:main",
            "hh4b_non_res_res_draw_hists = cli.draw_hists_non_res_res:main",
            "hh4b_submit = cli.submit_jobs:main",
            "hh4b_dump = cli.dump:main",
        ]
    },
    zip_safe=False,
)
