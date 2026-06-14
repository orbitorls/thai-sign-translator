import torch

from tsl.data.islr import ISLRDataset


def _write_clip(path, n_frames=2):
    import pandas as pd
    type_counts = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}
    rows = []
    for f in range(n_frames):
        for ltype, count in type_counts.items():
            for li in range(count):
                rows.append({
                    "frame": f, "type": ltype, "landmark_index": li,
                    "x": 1.0, "y": 2.0, "z": 3.0,
                })
    pd.DataFrame(rows).to_parquet(str(path), engine="pyarrow")


def _make_dataset_dir(tmp_path, signs):
    import pandas as pd
    pq_dir = tmp_path / "train_landmark_files"
    pq_dir.mkdir()
    rows = []
    for i, sign in enumerate(signs):
        rel = f"{i}.parquet"
        _write_clip(pq_dir / rel)
        rows.append({"path": f"train_landmark_files/{rel}", "sign": sign})
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(str(csv_path), index=False)
    return str(tmp_path), str(csv_path)


def test_dataset_len_classes_and_sample(tmp_path):
    root, csv = _make_dataset_dir(tmp_path, ["hello", "thanks", "hello"])
    ds = ISLRDataset(parquet_dir=root, csv_path=csv)
    assert len(ds) == 3
    assert ds.num_classes == 2
    assert sorted(ds.label_names) == ["hello", "thanks"]
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, int)
    assert x.ndim == 2
    assert x.shape[0] == 2
    assert x.dtype == torch.float32
    assert 0 <= y < ds.num_classes
    assert ds.label_names[y] == "hello"


def test_dataset_class_subset(tmp_path):
    root, csv = _make_dataset_dir(tmp_path, ["a", "b", "c", "a", "b", "c"])
    ds = ISLRDataset(parquet_dir=root, csv_path=csv, classes=["a", "c"])
    assert len(ds) == 4
    assert ds.num_classes == 2
    assert sorted(ds.label_names) == ["a", "c"]
    for i in range(len(ds)):
        _, y = ds[i]
        assert 0 <= y < 2
