import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replicator_to_coco import asset_root, convert  # noqa: E402


def test_asset_root():
    assert asset_root("/World/obj_07_keepsake_box/lid/render") == "/World/obj_07_keepsake_box"
    assert asset_root("/World/obj_01_bowl/render") == "/World/obj_01_bowl"


def test_convert_merges_parts_and_skips_background(tmp_path):
    inst = np.zeros((40, 60), np.uint16)
    inst[5:15, 5:15] = 10   # mug
    inst[20:30, 30:40] = 11  # box base
    inst[15:20, 30:40] = 12  # box lid
    Image.fromarray(inst).save(tmp_path / "instance_segmentation_0000.png")
    Image.new("RGB", (60, 40)).save(tmp_path / "rgb_0000.png")
    (tmp_path / "instance_segmentation_mapping_0000.json").write_text(json.dumps({
        "0": "BACKGROUND", "10": "/World/obj_00_mug/render",
        "11": "/World/obj_01_box/base/render", "12": "/World/obj_01_box/lid/render"}))
    (tmp_path / "instance_segmentation_semantics_mapping_0000.json").write_text(json.dumps({
        "0": {"class": "BACKGROUND"}, "10": {"class": "mug"}, "11": {"class": "box"}, "12": {"class": "box"}}))
    coco = convert(tmp_path)
    assert len(coco["images"]) == 1
    assert sorted(c["name"] for c in coco["categories"]) == ["box", "mug"]
    anns = {a["attributes"]["prim"]: a for a in coco["annotations"]}
    assert set(anns) == {"/World/obj_00_mug", "/World/obj_01_box"}
    box = anns["/World/obj_01_box"]
    assert box["area"] == 150.0  # base + lid merged
    assert box["bbox"] == [30.0, 15.0, 10.0, 15.0]
