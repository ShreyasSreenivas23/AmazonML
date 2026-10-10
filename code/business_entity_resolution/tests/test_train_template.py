"""End-to-end: template trains, writes heartbeat + OOF, and resumes after a crash. Needs torch (CPU ok)."""
import json

import pytest

torch = pytest.importorskip("torch")

ARGS = ["--cfg", "configs/example.yaml", "exp_id=T1", "train.epochs=3", "cv.n_splits=2"]


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    for var in ("RUNS_DIR", "OOF_DIR", "CKPT_DIR"):
        monkeypatch.setenv(var, str(tmp_path / var.lower()))
    monkeypatch.setenv("RUN_ID", "t-run")
    return tmp_path


def test_template_crash_then_resume(dirs, capsys, monkeypatch):
    import src.train_template as tt

    real_save = tt.save_ckpt

    def crash_after_first_save(path, **state):
        real_save(path, **state)
        raise RuntimeError("simulated crash after epoch 0")

    monkeypatch.setattr(tt, "save_ckpt", crash_after_first_save)
    with pytest.raises(RuntimeError):
        tt.main(ARGS)
    hb = json.loads((dirs / "runs_dir" / "t-run" / "heartbeat.json").read_text())
    assert hb["status"] == "failed" and "simulated crash" in hb["error"]

    monkeypatch.setattr(tt, "save_ckpt", real_save)
    capsys.readouterr()
    cv = tt.main(ARGS)
    assert "fold 0: resumed at epoch 1" in capsys.readouterr().out
    assert 0 < cv < 200
    hb = json.loads((dirs / "runs_dir" / "t-run" / "heartbeat.json").read_text())
    assert hb["status"] == "completed" and hb["best_metric"] == pytest.approx(cv)
    assert (dirs / "oof_dir" / "T1" / "oof.parquet").exists()
