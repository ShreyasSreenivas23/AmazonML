"""Entity-resolution output writer/validator against a tiny synthetic test split."""
import pytest

from src.er_submission import load_ground_truth, load_sources, main, validate_outputs, write_outputs


@pytest.fixture
def test_dir(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    rows = {
        1: ["S1-00001\tAcme Corp\t12 Main St, Springfield\tUS", "S1-00002\tChez Paul SARL\t3 Rue de Rivoli, Paris\tFrance",
            "S1-00003\tSolo Traders\tNear SBI ATM, Pune\tIndia"],
        2: ["S2-00047\tACME Corporation\t12 Main Street\tUS", "S2-00193\tAcme Inc\t9 Elm Rd\tUS"],
        3: ["S3-00812\tAcme Corp.\tMain St 12, Springfield\tUS", "S3-00004\tChez Paul\tRue de Rivoli 3\tFrance"],
    }
    for k, lines in rows.items():
        (d / f"test_source{k}.tsv").write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return d


def _write(p, header, lines):
    p.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_load_sources_keeps_france_and_strings(test_dir):
    s1, s2, s3 = load_sources(test_dir)
    assert len(s1) == 3 and "France" in set(s1["country"]) and s2["entity_id"].iloc[0] == "S2-00047"


def test_write_then_validate_roundtrip(test_dir, tmp_path):
    s1, _, _ = load_sources(test_dir)
    matches = {"S1-00001": ["S3-00812", "S2-00047", "S2-00047"], "S1-00002": ["S3-00004"]}
    cands = {"S1-00001": ["S2-00047", "S2-00193", "S3-00812"], "S1-00002": ["S3-00004"]}
    rep = write_outputs(matches, cands, s1["entity_id"], tmp_path / "out", test_dir=test_dir)
    assert rep.ok
    text = (tmp_path / "out" / "matching_results.tsv").read_text(encoding="utf-8")
    assert text.splitlines() == ["source1_entity_id\tmatched_entity_ids", "S1-00001\tS2-00047,S3-00812",
                                 "S1-00002\tS3-00004", "S1-00003\t"]


def test_write_refuses_match_outside_candidates(test_dir, tmp_path):
    s1, _, _ = load_sources(test_dir)
    with pytest.raises(ValueError):
        write_outputs({"S1-00001": ["S2-00193"]}, {"S1-00001": ["S2-00047"]}, s1["entity_id"], tmp_path / "o", test_dir)


@pytest.mark.parametrize("lines,needle", [
    (["S1-00001\tS2-00047", "S1-00002\t"], "missing"),                                 # S1-00003 missing
    (["S1-00001\tS2-00047", "S1-00001\t", "S1-00002\t", "S1-00003\t"], "duplicate source1"),
    (["S1-00001\tS2-00047,S2-00047", "S1-00002\t", "S1-00003\t"], "duplicate ids"),
    (["S1-00001\tS1-00002", "S1-00002\t", "S1-00003\t"], "not S2-/S3-"),
    (["S1-00001\tS2-99999", "S1-00002\t", "S1-00003\t"], "don't exist"),
    (["S1-00001\tS2-00047, S3-00812", "S1-00002\t", "S1-00003\t"], "spaces"),
    (["S1-00001,S2-00047", "S1-00002\t", "S1-00003\t"], "2 tab-separated"),            # comma instead of tab
])
def test_validator_catches_rejections(test_dir, tmp_path, lines, needle):
    p = _write(tmp_path / "m.tsv", "source1_entity_id\tmatched_entity_ids", lines)
    rep = validate_outputs(p, test_dir=test_dir)
    assert not rep.ok and any(needle in e for e in rep.errors), rep.errors


def test_wrong_header_and_cli(test_dir, tmp_path):
    p = _write(tmp_path / "m.tsv", "source1_entity_id\tmatches", ["S1-00001\t", "S1-00002\t", "S1-00003\t"])
    assert main(["validate", "--matching", str(p), "--test-dir", str(test_dir)]) == 1
    ok = _write(tmp_path / "ok.tsv", "source1_entity_id\tmatched_entity_ids", ["S1-00001\t", "S1-00002\t", "S1-00003\t"])
    assert main(["validate", "--matching", str(ok), "--test-dir", str(test_dir)]) == 0


def test_load_ground_truth(tmp_path):
    p = _write(tmp_path / "gt.tsv", "source1_entity_id\tmatched_entity_ids", ["S1-1\tS2-1,S3-2", "S1-2\t"])
    assert load_ground_truth(p) == {"S1-1": {"S2-1", "S3-2"}, "S1-2": set()}
