"""Editable install: ``baselines`` (under ``scripts/baselines``) and root ``utils``."""
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).resolve().parent

packages = ["baselines"]
package_dir = {"baselines": "scripts/baselines"}
py_modules = []

if (_ROOT / "utils.py").is_file():
    py_modules.append("utils")
elif (_ROOT / "utils").is_dir() and (_ROOT / "utils" / "__init__.py").is_file():
    packages.append("utils")

setup(
    packages=packages,
    package_dir=package_dir,
    py_modules=py_modules,
)
