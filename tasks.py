import sys
from invoke import task


@task
def benchmark(c):
    """CI benchmark"""
    c.run("pytest --benchmark-only")


@task
def black(c):
    c.run(f"{sys.executable} -m black --check src/ example/")


@task
def flake8(c):
    c.run(f"{sys.executable} -m flake8 -v")


@task
def mypy(c):
    c.run(f"{sys.executable} -m mypy src/ example/")


@task
def bandit(c):
    c.run(f"{sys.executable} -m bandit -c pyproject.toml -r src/")


@task
def lint(c):
    """Lint"""
    black(c)
    flake8(c)
    mypy(c)
    bandit(c)


@task(optional=["verbose", "keyword", "cov-lines"])
def test(c, verbose=False, keyword="", cov_lines=False):
    """Test without benchmark"""
    c.run(
        "pytest --benchmark-skip"
        + (" -vv" if verbose else "")
        + (f" -k{keyword}" if keyword else "")
        + (" --cov-report term-missing" if cov_lines else "")
    )


@task
def docbuild(c):
    """Sphinx build"""
    result = c.run(f"{sys.executable} -m sphinx -W -b html docs _build")
    if result.exited != 0:
        raise SystemExit(result.exited)


@task
def doc(c):
    """Sphinx autobuild"""
    result = c.run(f"{sys.executable} -m sphinx_autobuild -W -b html docs _build")
    if result.exited != 0:
        raise SystemExit(result.exited)
