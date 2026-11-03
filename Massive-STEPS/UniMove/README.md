# UniMove🚶‍♂️🏙️

This is the official PyTorch implementation for our paper, **UniMove: A Unified Model for Multi-city Human Mobility Prediction**, accepted at ACM SIGSPATIAL 2025. 

## ⚙️ Installation
### Environment
- Tested OS: Linux
- Python >= 3.11
- torch == 2.0.1
- CUDA == 11.7

### Dependencies:
1. Install Pytorch with the correct CUDA version.
2. Use the `pip install -r requirements.txt` command to install all of the Python modules and packages used in this project.


## ⚖ Repo Structure

```
UniMove  
├─location_feature                  # Location features [N,31], N means number of locations, 31=28(poi feature)+2(longitude & latitude)+1(popularity rank)  
│   ├─vocab_shanghai.npy        
│   ├─vocab_nanchang.npy      
│   └─vocab_lasa.npy    
│  
├─traj_dataset                      # The dataset examples where each trajectory is formatted as [user_id location,weekday,time;location,weekday,time;...].  
│   └─mini  
│       ├─test        
│       ├─train       
│       └─val     
│  
├─dataloader.py  
├─location_tower_model.py          # The architecture of location tower  
├─model.py                         # The architecture of trajectory tower  
├─utils.py                         # Train and evaluate methods  
├─main.py  
└─requirements.txt
```



## 🏃 Model Training
You can train UniMove with multi-city datasets and test with nanchang dataset as the following examples:

```python
python main.py --device cuda:0 --city nanchang shanghai lasa --target_city nanchang
```

Once your model is trained, you will find the logs recording the training process in the  `./logs_{args.city}` directory.

## 📜 Citation
If you find our work or this repository useful for your research, please consider citing our paper:
```
@article{han2025unimove,
  title={UniMove: A Unified Model for Multi-city Human Mobility Prediction},
  author={Han, Chonghua and Yuan, Yuan and Liu, Yukun and Ding, Jingtao and Feng, Jie and Li, Yong},
  journal={arXiv preprint arXiv:2508.06986},
  year={2025}
}g
```
