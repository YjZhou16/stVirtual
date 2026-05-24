from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py as _build_py


ROOT = Path(__file__).parent
SRC = ROOT / "src"
CORE_PACKAGES = {"model", "utils"}


def collect_extensions():
    extensions = []
    for package in sorted(CORE_PACKAGES):
        package_dir = SRC / package
        for py_file in sorted(package_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            module_name = f"{package}.{py_file.stem}"
            extensions.append(Extension(module_name, [str(py_file)]))
    return extensions


class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package in CORE_PACKAGES:
            return [module for module in modules if module[1] == "__init__"]
        return modules


setup(
    name="stvirtual",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=cythonize(collect_extensions(), compiler_directives={"language_level": "3"}),
    cmdclass={"build_py": build_py},
    install_requires=[
        "numpy==2.1.2",
        "pandas==2.3.2",
        "scanpy==1.11.4",
        "anndata==0.12.2",
        "torch==2.8.0",
        "scipy==1.16.2",
        "scikit-learn==1.7.2",
        "tqdm==4.67.1",
    ],
    zip_safe=False,
)
