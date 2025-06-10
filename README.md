## Data format

```bash
├── exp_name
│   ├── imgs
│   │   ├── 00.png
│   │   ├── 01.png
│   │   ├── ...
│   └── hand_tum.txt (MUST | Hand to base transformations, in TUM format)
│   └── intrinsics.txt (OPTIONAL | The 3x3 intrinsic matrix of the camera)
```



## Setup
1. Construct a conda env.
2. Follow the instructions in mast3r (`faiss-gpu-cu12`, no need to install optional)
3. for pycolmap, maybe you can install with conda
4. plotly pycolmap kapture kapture-localization seaborn


https://pypi.tuna.tsinghua.edu.cn/simple


## Correspondences


## FLow of Codes

`model_api.py`:
`pairs`: list
    `pairs[i]`: tuple of size 2
    `pairs[i][0/1]`: dict: "img" (1, 3, 384, 512), "true_shape", "idx", "instance"

`mapping.py`: 
line 76 `images` is a list, each element is a dict contains "img" (1, 3, 384, 512) , "true_shape" ([[384, 512]]), "to_orig" 3x3 matrix being diagonal (1.25,1.25,1.0) (640 / 1.25 = 512), "idx", "instance" (e.g. "0_Color.png"), "orig_shape" 
line 79 `matching_pairs` contains $n(n+1)/2$ elements from `images`
line 96-105 
    `output`: 
        "view1": quite similar to `images`, but the batch size is 4, "img" (4, 3, 384, 512)
        "view2": the corresponding pair for "view1"
        "pred1" and "pred2": "pts3d" (4, 384, 512, 3), "conf" (4, 384, 512), "desc" (4, 384, 512, 24), "desc_conf" (4, 384, 512)
    `im_images_chunk`: dict, keys are something like (0,1), (1,2) etc. Each elements contains (n, 2) matches 

