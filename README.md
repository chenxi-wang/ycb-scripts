# ycb-scripts
Scripts to generate and process point clouds in [YCB dataset](http://www.ycbbenchmarks.com/).

## Usage

### Download Dataset
```bash
    python ycb_downloader.py
```

### Generate Clouds From RGB-D Images

```bash
    python ycb_generate_point_cloud.py
```

### Transform Clouds to Unified Camera Coordinate System
```bash
    python ycb_transform_point_cloud.py
```

### Denoise Transformed Clouds
```bash
    python ycb_filter_point_cloud.py
```

After processing, all point clouds are denoised and transformed to the coordinate system of camera "NP5". It is helpful for object grasp pose generation.
