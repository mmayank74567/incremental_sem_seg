
# <div align="center">Few-shot Incremental Learning for Semantic Segmentation</div>

<font size = "3">**Mayank Mishra, Anirban Chakraborty, R. Venkatesh Babu**</font>

* ### Generating masks from ViT
A sample command to create saliency masks using the ViT-based method is given below. 
```
python creating_sal_maps.py --output_dir sal_maps_fromvit --log_name creating_new_sal_map_from_pretrained_dino --arch vit_small --patch_size 8 --temp_output_dir temp_images --root_image_dir data_root/VOC2012 --gpu_id 0
```

* ### Datasets
   The dataset used can be downloaded from the following links:
	   <ul>
	   <li>[PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)</li>
	   </ul>
Structure for the dataset directory
```
data_root/
    --- VOC2012/
        --- Annotations/
        --- ImageSet/
        --- JPEGImages/
        --- SegmentationClassAug/
        --- saliency_map/
```

* ### Training
A sample command to run ViT-UL on task {10-1} with saliency map from ViT is given below. (``--saliency_map`` should direct to the directoty containing saliency maps from ViT)
```
python main.py --data_root data_root/VOC2012 --model deeplabv3_resnet101 --gpu_id 0 --crop_val --lr 0.01 --batch_size 32 --train_epoch 50 --loss_type bce_loss --dataset voc --task 10-1 --overlap --lr_policy poly --pseudo --freeze --bn_freeze --unknown --w_transfer --amp --mem_size 0 --log_name SSUL_VOC_10-1-sal_maps_fromvit --saliency_map sal_maps_fromvit --exp_str exp_0
```
## Acknowledgement
Our implementation is based on [SSUL](https://github.com/clovaai/SSUL) and [DINO](https://github.com/facebookresearch/dino) repositories.
