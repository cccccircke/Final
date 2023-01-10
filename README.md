# Final

## Methodology 
* 在缺少測量值的情況下計算有條件的產品故障率，並將其與無條件產品故障率 0.212608 進行比較。
* 在缺少測量值的情況下計算有條件的產品故障率，並將其與無條件產品故障率 0.212608 進行比較。
* 當 abs(z) > 2.5 且 p-value < 2 % 時，缺失 measurement_3 和缺失 measurement_5 的條件故障率明顯偏離平均故障率，可以在模型中使用特徵missing_3 和 missing_5
* 特徵工程：通過添加按產品代碼分組的測量聚合統計數據作為新特徵，將產品代碼用於特徵工程
* 主要NA用HuberRegressor處理
* 其他NA用KNNImputer處理
* Model architecture用GroupKFold(n_splits=5)拆分再用linear_model.LogisticRegression預測

## Pre-trained Models
You can download pretrained models here:

- [My model link](https://github.com/cccccircke/Final/tree/master/model) trained on linear_model.LogisticRegression 
>📋 將下載後的pickle檔位置放到要load model的路徑(參閱重現步驟的[step3](#3)中的第二格)
## Requirements
1. 更改路徑(在[重現步驟](#re)裡指示)
2. 安裝feature_engine(執行的ipynb檔裡都有寫，理論上可以不動，怕有問題還是放上來)
 ```train
 ! pip install feature_engine
 ```
## Training
請看重現步驟的[step2](#2)

<a name="re"/>

## 重現步驟
* step1：直接git整個專案 或[點選此連結](https://github.com/cccccircke/Final/tree/master/model)自行下載model
```git
git clone https://github.com/cccccircke/Final.git
```
<a name="2"/>

* step2(若要直接使用model可以跳過此步驟)：
  * 開啟109550106_Final_train.ipynb檔案，將檔案中的這三行程式(如下)更改為自己的檔案的路徑
  * 在109550106_Final_train.ipynb中的第三個程式格
  ```chang
  df_train = pd.read_csv('/content/drive/MyDrive/tabular-playground-series-aug-2022/train.csv')# read data
  df_test = pd.read_csv('/content/drive/MyDrive/tabular-playground-series-aug-2022/test.csv')
  submission = pd.read_csv('/content/drive/MyDrive/tabular-playground-series-aug-2022/sample_submission.csv')
  ```
  * 在檔案中的最後一格中，可以更改pickle檔案load的位置
  ```load
  with open('./x.pickle', 'wb') as f:
    pickle.dump(model, f)
  ```
<a name="3"/>
  
* step3：
  * 開啟109550106_Final_inference.ipynb檔案，將檔案中的這三行程式(如下)更改為自己的檔案的路徑
  * 在109550106_Final_inference.ipynb中的第三個程式格
  ```chang
  df_train = pd.read_csv('/content/drive/MyDrive/tabular-playground-series-aug-2022/train.csv')# read data
  df_test = pd.read_csv('/content/drive/MyDrive/tabular-playground-series-aug-2022/test.csv')
  submission = pd.read_csv('/content/drive/MyDrive/tabular-playground-series-aug-2022/sample_submission.csv')
  ```
  * 在檔案中的最後一格中，將這兩行程式(如下)更改為自己存儲的pickle檔案的路徑，load到 Model中
  ```load
  with open('/content/drive/MyDrive/tabular-playground-series-aug-2022/x.pickle', 'rb') as f:
    Model = pickle.load(f)
  ```
* step4：執行完在現行檔案下會看到109550106.csv檔(要更改路徑可以調整在109550106_Final_inference.ipynb的最後一格的程式(以下))
  ```load
  submission.to_csv(f"./109550106.csv", index=False)
  ```

## Results

![image](https://i.imgur.com/GzSdvXO.png)


