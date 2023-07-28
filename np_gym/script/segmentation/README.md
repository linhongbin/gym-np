# share models

rgb scene: https://drive.google.com/file/d/1YvqtTy2EkBVPBurV8H7opdA5MmgtnkVv/view?usp=share_link

# Install

```sh
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install fiftyone brotli labelme
```


# reference
https://blog.roboflow.com/detectron2-custom-instance-segmentation/
https://github.com/TannerGilbert/Detectron2-Train-a-Instance-Segmentation-Model


# Procedure

1. collect data
   ```sh
   python segmentation/collect_data.py -s 111
   ```
2. labelme
3. label2coco
```sh
python segmentation/labelme2coco.py --indir ./data/segmentation/ambf/lableme/ --output ./data/segmentation/ambf/train.json
python segmentation/labelme2coco.py --indir ./data/segmentation/ambf/lableme/ --output ./data/segmentation/ambf/test.json
```
4. run notebook
5. copy
   ```sh
   cp segmentation/output/model_final.pth ./gym_np/model/segment.pth 
   ```