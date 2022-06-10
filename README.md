# AICUP_STAS_Segmentation

# 介紹:
這次aicup比賽得到Private第七的分數，在這裡分享我程式碼，我的最佳模型為三個模型的voting ensemble，當初在訓練的時候，我是三個模型**各自訓練**，並使用PIL與numpy函式庫來進行ensemble的處理，所以我這裡會分享三份Jupyter notebook與三份模型，並額外付上voting ensemble的程式。

# 環境與套包

```
pip install monai
pip install xxx
pip install xxx
```

# 模型與權重

類別|模型名稱|Jupyter Notebook|權重檔|
--|--|--|--|
模型一|DeepLabV3Plus + tf_efficientnetv2_m_in21ft1k|--1-|[tf_efficientnetv2_m_in21ft1k.pth](https://drive.google.com/file/d/1R8ez_bH2H5KsshnWdeA4rcYTcUcqbHhD/view?usp=sharing)|
模型二|DeepLabV3Plus + tu-eca_nfnet_l2|--2-|[tu-eca_nfnet_l2_DeepLabV3Plus.pth](https://drive.google.com/file/d/1Cbgkb0SNsghGo8x0SgHgYPR9kAbOJjLA/view?usp=sharing)|
模型三|DeepLabV3Plus + tu-tf_efficientnet_b6_ns|--3-|[tu-tf_efficientnet_b6_ns.pth](https://drive.google.com/file/d/1lkkzq2SbDvxgvNDKGoGMiRDNEZ7399Cm/view?usp=sharing)|
Ensemble|Voting Ensemble|--4-|-|

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
