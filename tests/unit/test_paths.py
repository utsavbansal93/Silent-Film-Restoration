import os

from pipeline.common.paths import env, project_root, runs_dir


def test_env_default():
    os.environ.pop("SILENT_FILM_ENV", None)
    assert env() == "local"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SILENT_FILM_ENV", "colab")
    assert env() == "colab"


def test_project_root_exists():
    assert project_root().is_dir()


def test_runs_dir_created():
    d = runs_dir()
    assert d.is_dir() and d.name == "runs"
