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


def test_annotator_boxes_are_authoritative(tmp_path):
    """An object missing from the instance mask still gets a box; parts merge; disagreeing masks are dropped."""
    inst = np.zeros((40, 60), np.uint16)
    inst[5:15, 5:15] = 10  # mug, agrees with its box
    inst[30:32, 50:52] = 11  # bowl mask shrunk to a few pixels: disagrees with its box
    Image.fromarray(inst).save(tmp_path / "instance_segmentation_0000.png")
    Image.new("RGB", (60, 40)).save(tmp_path / "rgb_0000.png")
    (tmp_path / "instance_segmentation_mapping_0000.json").write_text(json.dumps({
        "10": "/World/obj_00_mug/render", "11": "/World/obj_02_bowl/render"}))
    (tmp_path / "instance_segmentation_semantics_mapping_0000.json").write_text(json.dumps({
        "10": {"class": "mug"}, "11": {"class": "bowl"}}))
    rows = np.array([(0, 5, 5, 14, 14, 0.0), (1, 30, 15, 39, 19, 0.0), (1, 30, 20, 39, 29, 0.0),
                     (2, 40, 22, 55, 35, 0.0)],
                    dtype=[("semanticId", "<u4"), ("x_min", "<i4"), ("y_min", "<i4"), ("x_max", "<i4"),
                           ("y_max", "<i4"), ("occlusionRatio", "<f4")])
    np.save(tmp_path / "bounding_box_2d_tight_0000.npy", rows)
    (tmp_path / "bounding_box_2d_tight_labels_0000.json").write_text(json.dumps({
        "0": {"class": "mug"}, "1": {"class": "box"}, "2": {"class": "bowl"}}))
    anns = {}
    coco = convert(tmp_path)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    for a in coco["annotations"]:
        anns[cats[a["category_id"]]] = a
    assert set(anns) == {"mug", "box", "bowl"}
    assert anns["box"]["bbox"] == [30.0, 15.0, 10.0, 15.0]  # lid + base rows merged
    assert anns["mug"]["segmentation"] and anns["mug"]["attributes"]["prim"] == "/World/obj_00_mug"
    assert anns["bowl"]["segmentation"] == []  # box kept, bad mask dropped
