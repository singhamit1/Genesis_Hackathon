from src.data.load import load_datasets

def test_load_datasets_handles_missing(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    out = load_datasets(d)
    assert set(out.keys()) >= {"logs","topology","config","app_usage","user_activity","external"}
