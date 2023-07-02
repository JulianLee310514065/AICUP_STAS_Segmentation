# AICUP_STAS_Segmentation
> 深度學習課程期中考[(Link)](https://tbrain.trendmicro.com.tw/Competitions/Details/22)

此比賽為AUCUP 2022 肺腺癌病理切片影像之腫瘤氣道擴散偵測競賽 II：運用影像分割作法於切割STAS輪廓之程式碼上傳區。

# 介紹:
這次aicup比賽得到Private第七的分數，在這裡分享我程式碼，我的最佳模型為三個模型的Voting Ensemble，當初在訓練的時候，我是三個模型**各自訓練**，並使用PIL與numpy函式庫來進行Ensemble的處理，所以我這裡會分享三份Jupyter Notebook與三份模型，並額外付上Voting Ensemble的程式，然後如果程式使用上有什麼問題或Bug，也歡迎在Issues區留言提問。

# Leaderboard
|Public Score|Public Rank|Public Score|Private Rank|
|--|--|--|--|
|0.898982|4 / 307|0.916327|7 / 307|
---

## 更新
最後結果: 

**第十四名**

<div align="center">
	<img src="https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/%E9%A0%81%E9%9D%A2%E6%93%B7%E5%8F%96%E8%87%AA-%E8%82%BA%E8%85%BA%E7%99%8C%E7%AB%B6%E8%B3%BDII-%E6%95%99%E8%82%B2%E9%83%A8%E5%85%A8%E5%9C%8B%E5%A4%A7%E5%B0%88%E6%A0%A1%E9%99%A2%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E7%AB%B6%E8%B3%BD(AI%20CUP%202022)%20.jpg" alt="final" width="600">
</div>


# 環境與套包
訓練環境
```
TWCC : cm.2xsuper (TWCC 4 GPU + 16 cores + 240GB memory + 120GB share memory)
映像檔 : pytorch-22.02-py3:latest
```
套包
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
權重檔皆存放於Google雲端，可自行下載使用

類別|模型名稱/用途|Jupyter Notebook|權重檔|模型預測結果|
--|--|--|--|--|
Label前處理|json to png|[DataPreprocess.ipynb](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/DataPreprocess.ipynb)|-|-|
模型一|DeepLabV3Plus + tf_efficientnetv2_m_in21ft1k|[tf_efficientnetv2_m_in21ft1k.ipynb](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/tf_efficientnetv2_m_in21ft1k.ipynb)|[tf_efficientnetv2_m_in21ft1k.pth](https://drive.google.com/file/d/1R8ez_bH2H5KsshnWdeA4rcYTcUcqbHhD/view?usp=sharing)|[模型一結果](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/Result/tf_efficientnetv2_m_in21ft1k_origin.zip)|
模型二|DeepLabV3Plus + tu-eca_nfnet_l2|[tu-eca_nfnet_l2_DeepLabV3Plus.ipynb](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/tu-eca_nfnet_l2_DeepLabV3Plus.ipynb)|[tu-eca_nfnet_l2_DeepLabV3Plus.pth](https://drive.google.com/file/d/1Cbgkb0SNsghGo8x0SgHgYPR9kAbOJjLA/view?usp=sharing)|[模型二結果](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/Result/tu-eca_nfnet_l2_DeepLabV3Plus_origin.zip)|
模型三|DeepLabV3Plus + tu-tf_efficientnet_b6_ns|[tu-tf_efficientnet_b6_ns.ipynb](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/tu-tf_efficientnet_b6_ns.ipynb)|[tu-tf_efficientnet_b6_ns.pth](https://drive.google.com/file/d/1lkkzq2SbDvxgvNDKGoGMiRDNEZ7399Cm/view?usp=sharing)|[模型三結果](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/Result/tu-tf_efficientnet_b6_nsorigin.zip)
Ensemble|Voting Ensemble|[Image_ensemble.ipynb](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/Image_ensemble.ipynb)|-|[最後結果](https://github.com/JulianLee310514065/AICUP_STAS_Segmentation/blob/main/Result/Finall%20ensemble.zip)|

# 使用說明

### 前處理
需修改資料夾，資料夾內需包含LabelMe之.json檔

```
folder_path = "{YOUR PATH}"

# 如 : 
folder_path = "SEG_Train_Datasets/Train_Annotations/"
os.listdir(folder_path)[:5]
```

### 模型訓練以及驗證
需修改訓練圖片之資料夾
```
data_path = "{YOUR PATH}"
```

如下之`SEG_Train_Datasets`，資料夾內需放子資料夾`Train_Images`及`Train_Annotations_png`，前者存放訓練Image後者存放訓練Mask
```
data_path = './SEG_Train_Datasets/'

SEG_Train_Datasets
        ↳ Train_Images
        ↳ Train_Annotations_png
```
>

改驗證圖片路徑
```
tempdir = "{YOUR PATH}"  

如:
tempdir = "./Pravite_Image1/"
```

驗證時除了需確認驗證圖片的檔案位置，還須修改權重路徑到你權重下載的位置

```
model.load_state_dict(torch.load("{YOUR PATH}"))
```



>

此外因為我三個模型各自的預測結果圖都是預設存到同目錄底下`./Predict`資料夾，所以同時跑三個模型時輸出會被蓋掉，可在模型輸出處做修改
```python
saverPD = SaveImage(output_dir="{YOUR PATH}", output_ext=".png", output_postfix=f"{Pub_data[i].split('/')[-1].split('.')[0]}",scale=255,separate_folder=False)
saverPD(test_outputs[0].cpu())
```


### Ensemble Code
Ensemble時也需要注意三個的檔案位置

```python
path1 = "{Predict Path1}"
path2 = "{Predict Path2}"
path3 = "{Predict Path3}"
```

此外還要注意的是做Ensemble的照片通道數需統一，即 (1716, 942, 1) 或 (1716, 942, 3)，如不是則必須修改程式，建議全部改為單一通道。

1 Channel
```python
img1 = Image.open(os.path.join(path1, filename))
img1_ar = np.asarray(img1)
img1_ar = np.where(img1_ar > 0, 1, 0)

```

3 or more Channel
```python
img1 = Image.open(os.path.join(path1, filename))
img1_ar = np.asarray(img1)
img1_ar = np.where(img1_ar[:, :, 0] > 0, 1, 0)

```

# Run on Colab
程式範例展示，因為Colab的環境與GPU數量與當初訓練使用時不同，最後結果也會不同，請勿使用此Colab檔案來驗證權重及輸出等，若要驗證權重或模型等資訊，請使用**相同的環境**。

[Colab](https://colab.research.google.com/drive/1uStw2mKnpq3j8E4Tpf7qy8zx2QAVY1nZ?usp=sharing)


