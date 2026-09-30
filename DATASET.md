# CoPaint datasets: PaintStatic & PaintDynamic

Research-use only; access on request (see the Data Availability Statement in the paper).
Default configs expect the dataset at `/workspace/CoPaint_datasets/`; change the paths in `configs/training/*` otherwise.

```
CoPaint_datasets/
  PaintStatic/
    images/00000.jpg … 03165.jpg
    annotations/captions.json             [{"image", "text": "[s0:Sketch] <caption>"}]
    annotations/metadata.jsonl            {"file_name", "text": "[stage0] sketch, <caption>"}   (HF imagefolder)
    annotations/paintstatic_metadata.csv  image, stage_idx, stage, split, caption, ...
    annotations/splits/{train,val,test}.txt
  PaintDynamic/
    videos/<videoid>.mp4
    frames/<clip_id>/00-15.jpg            16 frames of each clip in t2v_clips.jsonl
    annotations/annotation_ctrl.csv       clip_id, videoid, prompt   (one prompt per clip)
    annotations/t2v_clips.jsonl           {"clip_id", "videoid", "sampled_indices"[16], "stage_idx"[16], ...}
    annotations/sketchctrl_clips.jsonl    {"video_path", "sketch_index", "target_indices"}
    annotations/rgbctrl_clips.jsonl       {"video_path", "rgb_index", "target_indices"}
    annotations/frame_stage_labels.json   {videoid: {"frames": [{"frame", "frame_index", "stage_idx"}]}}
    annotations/splits/{train,val,test}.txt   video-level split (ids like 0304)
```
Stage ids: 0 sketch, 1 line art, 2 base color, 3 light/shading.

| Loader (`paintflow/data/dataset.py`) | Used by | Annotation |
|---|---|---|
| `PaintDataset` | `train.py` (`configs/training/v1/training.yaml`) | `t2v_clips.jsonl` + `annotation_ctrl.csv` |
| `PaintCtrlDataset` | `train_sparsectrl.py` (sketch), RGB variant | `sketchctrl_clips.jsonl` / `rgbctrl_clips.jsonl` |
| `PaintVideo` | `eval.py` (`--test_split .../splits/test.txt`) | `annotation_ctrl.csv` |
| `PaintStaticDataset` | stage-aware text encoder / VAE / U-Net pretraining | `captions.json` |

All loaders take an optional `split_file` (train/val/test).
