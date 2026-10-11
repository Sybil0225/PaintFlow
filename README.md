# PaintFlow and PaintCtrl

PaintFlow generates painting-process videos. PaintCtrl adds RGB and sketch conditioning for the CoPaint workflow.

## Model weights

Model access is restricted and granted on request for non-commercial research only. Commercial use is not permitted.

| Model | Checkpoint |
|---|---|
| PaintFlow | `paintflow.ckpt` |
| RGB control | `rgbctrl.ckpt` |
| Sketch control | `sketchctrl.ckpt` |

Place all three checkpoints in the repository root:

```text
PaintFlow/
  paintflow.ckpt
  rgbctrl.ckpt
  sketchctrl.ckpt
```

Update the checkpoint paths in the inference configuration and scripts. These checkpoints also require the base image model, tokenizer, text encoder, and VAE used by the inference pipeline.

## Data

[PaintStatic and PaintDynamic on Zenodo](https://zenodo.org/records/23032920)

Dataset structure and documentation are provided on Zenodo. Access is restricted to protect the copyrights of collaborating artists. Research and non-commercial use only.

## Code

- `animate.py`: painting-process generation.
- `animate_ctrl.py`: conditioned generation.
- `configs/`: training and inference configurations.
- `requirements.txt`: Python dependencies.

Adjust environment-specific paths and GPU settings before running.

## Authors

Yixuan Zhang, Fengzhou Liang, and Yuta Sugiura.
