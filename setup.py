from typing import List
import setuptools
import sys


class Lint(setuptools.Command):
    user_options: List[str] = []
    description = "Lint"

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess

        lint_steps = [
            "black --check h5grove/ example/ test/",
            "flake8 -v",
            "mypy h5grove/ example/ test/",
        ]

        for step in lint_steps:
            cmd = step.split()[0]
            print(f"Running {cmd}...")
            errno = subprocess.call([sys.executable, "-m", *step.split()])
            if errno != 0:
                raise SystemExit(errno)
            print(f"{cmd} check passed !")


class Doc(setuptools.Command):
    user_options: List[str] = []
    description = "Sphinx build"

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess

        cmd = "sphinx -W -b html docs _build"

        errno = subprocess.call([sys.executable, "-m", *cmd.split()])
        if errno != 0:
            raise SystemExit(errno)


if __name__ == "__main__":
    setuptools.setup(cmdclass={"lint": Lint, "doc": Doc})
