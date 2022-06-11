# AICUP_STAS_Segmentation

# 介紹:
這次aicup比賽得到Private第七的分數，在這裡分享我程式碼，我的最佳模型為三個模型的voting ensemble，當初在訓練的時候，我是三個模型**各自訓練**，並使用PIL與numpy函式庫來進行ensemble的處理，所以我這裡會分享三份Jupyter notebook與三份模型，並額外付上voting ensemble的程式。

# 環境與套包

```
pip install monai
pip install -U Setuptools
pip install git+https://github.com/qubvel/segmentation_models.pytorch
pip install adabelief-pytorch==0.2.0

#如果沒有PyTorch的話
pip install torch
pip install torchvision
```

### 如果遇到smp無法使用，有可能是Jupyter Notebook的問題，執行:

```
pip install ipywidgets widgetsnbextension
jupyter nbextension enable --py widgetsnbextension
```
### 如果是`import cv2`的問題，我自己在兩個不同的伺服器上遇到過

```
#第一種解法: 降低版本 (NVIDIA DLI Server可行)
pip install opencv-python==3.4.5.20

#第二種解法: 升級一些東西 (TWCC可行)
sudo apt update
sudo apt-get install libsm6 libxrender1 libfontconfig1 libgl1-mesa-glx
```

# 模型與權重

類別|模型名稱/用途|Jupyter Notebook|權重檔|模型預測結果|
--|--|--|--|--|
Label前處理|json to png|[DataPreprocess.ipynb](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/DataPreprocess.ipynb)|-|
模型一|DeepLabV3Plus + tf_efficientnetv2_m_in21ft1k|[tf_efficientnetv2_m_in21ft1k.ipynb](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/tf_efficientnetv2_m_in21ft1k.ipynb)|[tf_efficientnetv2_m_in21ft1k.pth](https://drive.google.com/file/d/1R8ez_bH2H5KsshnWdeA4rcYTcUcqbHhD/view?usp=sharing)|[模型一結果](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/Result/tf_efficientnetv2_m_in21ft1k_origin.zip)|
模型二|DeepLabV3Plus + tu-eca_nfnet_l2|[tu-eca_nfnet_l2_DeepLabV3Plus.ipynb](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/tu-eca_nfnet_l2_DeepLabV3Plus.ipynb)|[tu-eca_nfnet_l2_DeepLabV3Plus.pth](https://drive.google.com/file/d/1Cbgkb0SNsghGo8x0SgHgYPR9kAbOJjLA/view?usp=sharing)|[模型二結果](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/Result/tu-eca_nfnet_l2_DeepLabV3Plus_origin.zip)|
模型三|DeepLabV3Plus + tu-tf_efficientnet_b6_ns|[tu-tf_efficientnet_b6_ns.ipynb](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/tu-tf_efficientnet_b6_ns.ipynb)|[tu-tf_efficientnet_b6_ns.pth](https://drive.google.com/file/d/1lkkzq2SbDvxgvNDKGoGMiRDNEZ7399Cm/view?usp=sharing)|[模型三結果](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/Result/tu-tf_efficientnet_b6_nsorigin.zip)
Ensemble|Voting Ensemble|[Image_ensemble.ipynb](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/Image_ensemble.ipynb)|-|[最後結果](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/Result/Finall%20ensemble.zip)|

# 使用說明

### 模型訓練(train)
123 
牽著手
### 模型預測(test)
456
抬起頭
### Ensemble Code
789
我們私奔到月球
