import sys
from invoke import task


@task
def benchmark(c):
    """CI benchmark"""
    c.run("pytest --benchmark-only")


@task
def black(c):
    c.run(f"{sys.executable} -m black --check h5grove/ example/ test/")


@task
def flake8(c):
    c.run(f"{sys.executable} -m flake8 -v")


@task
def mypy(c):
    c.run(f"{sys.executable} -m mypy h5grove/ example/ test/")


@task
def lint(c):
    """Lint"""
    black(c)
    flake8(c)
    mypy(c)


@task
def test(c, verbose=False, keyword="", cov_lines=False):
    """Test without benchmark"""
    c.run(
        "pytest --benchmark-skip"
        + (" -vv" if verbose else "")
        + ((" -k" + keyword) if keyword else "")
        + (" --cov-report term-missing" if cov_lines else "")
    )


@task
def doc(c):
    """Sphinx build"""
    result = c.run(f"{sys.executable} -m sphinx -W -b html docs _build")
    if result.exited != 0:
        raise SystemExit(result.exited)
