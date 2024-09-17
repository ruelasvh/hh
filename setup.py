from setuptools import setup, find_packages

setup(
    name="hh",
    version="0.2",
    description="Code to plot the HH4b analysis",
    url="https://github.com/ruelasvh/hh.git",
    author="Victor Ruelas",
    author_email="victor.hugo.ruelas.rivera@cern.ch",
    license="MIT",
    packages=find_packages(include=["hh", "hh.*", "cli", "cli.*"]),
    python_requires=">=3.8",
    install_requires=[
        "awkward>=2.3.3",
        "matplotlib>=3.7.2",
        "mplhep>=0.3.28",
        "numpy>=1.26.1",
        "scipy>=1.11.2",
        "uproot>=5.0.11",
        "vector>=1.1.0",
        "coloredlogs>=15.0.1",
        "h5py>=3.9.0",
        "pandas>=2.1.0",
        "tables>=3.9.2",
        "htcondor>=23.0.6",
        "cabinetry>=0.6.0",
        "pyarrow>=17.0.0",
    ],
    entry_points={
        "console_scripts": [
            "hh4b_non_res_res_make_hists = cli.make_hists_non_res_res:main",
            "hh4b_non_res_res_draw_hists = cli.draw_hists_non_res_res:main",
            "hh4b_submit = cli.submit_jobs:main",
            "hh4b_dump = cli.dump:main",
            "hh4b_fit = cli.fit:main",
            "hh4b_merge = cli.merge_jobs:main",
        ]
    },
    zip_safe=False,
)
