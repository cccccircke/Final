# Final

## model link
假設已取得其它檔案：[只想得到model路徑請按此](https://github.com/cccccircke/Final/tree/master/model)
## Methodology 
* 在缺少測量值的情況下計算有條件的產品故障率，並將其與無條件產品故障率 0.212608 進行比較。
* 在缺少測量值的情況下計算有條件的產品故障率，並將其與無條件產品故障率 0.212608 進行比較。
* 當 abs(z) > 2.5 且 p-value < 2 % 時，缺失 measurement_3 和缺失 measurement_5 的條件故障率明顯偏離平均故障率，可以在模型中使用特徵missing_3 和 missing_5
* 特徵工程：我們可以通過添加按產品代碼分組的測量聚合統計數據作為新特徵，將產品代碼用於特徵工程
* 主要NA用HuberRegressor處理
* 其他NA用KNNImputer處理
* Model architecture用linear_model.LogisticRegression和GroupKFold(n_splits=5)預測


## 重現步驟
* step1：直接git整個專案 或[點選此連結](https://github.com/cccccircke/Final/tree/master/model)自行下載model
```git
git clone https://github.com/cccccircke/Final.git
```
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
>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
