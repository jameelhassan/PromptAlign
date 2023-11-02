## Run PromptAlign
We provide two bash scripts under `./scripts`. You can modify the paths and other args in the respective files in the scripts.     

To run PromptAlign on domain generalization datasets is shown below:
```
sh scripts/prompt_align/palign_dg.sh imagenet_a 1
```

For Fine Grained datasets:
```
sh scripts/prompt_align/palign_finegrain.sh ucf_101 1
```

### Config File Settings
We use the source data statistics computed on ImageNet for the alignment loss between tokens. Download the source data statistics 
for MaPLe from [here](https://drive.google.com/drive/folders/1ls9jWVFzlh-0t_O9dwxbH_IyCRpehQzK).

Create a folder ```output/features/``` and store the means and variances inside it. This path is specified in the config, this can 
be changed as required. All experiments output logs will be saved inside ```output/evaluation/```
