import pathlib
from setuptools import setup, find_packages


PKG_NAME = "psn"
VERSION = "0.1.0"
EXTRAS = {}


def _read_file(fname):
    with pathlib.Path(fname).open(encoding="utf-8") as fp:
        return fp.read()


def _read_install_requires():
    """Parse a pip-compile-generated requirements.txt without external deps."""
    out = []
    with pathlib.Path("requirements.txt").open() as fp:
        for line in fp:
            stripped = line.rstrip()
            # Skip blanks, indented lines (pip-compile annotations), comment-only lines
            if not stripped or stripped.startswith((" ", "\t", "#")):
                continue
            # Strip trailing inline comments
            req = stripped.split("#", 1)[0].strip()
            if req:
                out.append(req)
    return out


def _fill_extras(extras):
    if extras:
        extras["all"] = list(set([item for group in extras.values() for item in group]))
    return extras


setup(
    name=PKG_NAME,
    version=VERSION,
    author="Haochen Shi, Xingdi Yuan, Bang Liu",
    url="https://github.com/evolving-skill-networks/psn",
    description="Evolving Programmatic Skill Networks: lifelong-learning LLM agents with composable skills",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    project_urls={
        "Paper":        "https://arxiv.org/abs/2601.03509",
        "Project page": "https://evolving-skill-networks.github.io/",
        "Source":       "https://github.com/evolving-skill-networks/psn",
        "Issues":       "https://github.com/evolving-skill-networks/psn/issues",
    },
    keywords=[
        "Open-Ended Learning",
        "Lifelong Learning",
        "Embodied Agents",
        "Large Language Models",
    ],
    license="MIT License",
    packages=find_packages(include=["skillnet", "skillnet.*"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=_read_install_requires(),
    extras_require=_fill_extras(EXTRAS),
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3.10",
    ],
)
