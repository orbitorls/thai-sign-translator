import os
from tsl.eval.metrics import confusion_matrix_fig


def test_confusion_matrix_writes_png(tmp_path):
    out_png = str(tmp_path / "cm.png")
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    labels = ["hello", "thanks", "yes"]
    confusion_matrix_fig(y_true, y_pred, labels, out_png)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 0
    with open(out_png, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"
