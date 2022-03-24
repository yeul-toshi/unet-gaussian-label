# Experiment using CMC dataset

## Dataset

https://github.com/DeepPathology/MITOS_WSI_CMC  

## How to experiment

0. Build Environment  

```
docker pull continuumio/anaconda3:2021.05  
docker build -t gaussunet .  
docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -p 8888:8888 --name gaussunet gaussunet  
```

1. Clone the git repository of CMC dataset   
`git clone https://github.com/DeepPathology/MITOS_WSI_CMC.git`  
2. Open jupyter notebook  
`jupyter notebook --allow-root --ip 0.0.0.0`  
3. Download WSIs using MITOS_WSI_CMC/Setup.ipynb  
4. Make a dataset with Gaussian labels using make_dataset.ipynb  
5. Train the model  
`python train.py`  
6. Evaluate the model using evaluate.ipynb  