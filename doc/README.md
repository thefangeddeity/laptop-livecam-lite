# doc/

Two things this directory deliberately does not contain.

## Camera imagery

The proposal pipeline writes `median.png`, `camera_annotated.png`,
`floor_candidates.png`, `floor_quad.png` and `stability.png`, and reads a
phone reference photo. All of them are pictures of a family living space and
this repository is public, so they are gitignored and stay on the node that
produced them. Regenerate them from your own capture:

    python3 bin/lightcv-propose

`map.png` **is** tracked. It is a rendered top-down map -- polygons, labels
and tokens -- with no camera content in it.

## Detector weights

`cv/models/yolov8n.onnx` is gitignored. The export comes from Ultralytics,
whose code is AGPL-3.0 and whose weight licensing this project has not
resolved; shipping the file from a public GPL-3.0 repository would settle
that question by accident rather than by decision. Put your own export at
that path:

    mkdir -p cv/models
    cp /path/to/yolov8n.onnx cv/models/

Nothing else needs it: `lightcv-selftest` exercises the world model, the
floor homography and the renderer without a detector.
