import xgboost as xgb
import numpy as np

# データセットの作成
X = np.random.rand(10000, 50)
y = np.random.randint(0, 2, size=10000)

# データをDMatrix形式に変換
dtrain = xgb.DMatrix(X, label=y)

# GPU設定のパラメータ
params = {
    "tree_method": "hist",  # "gpu_hist" の代わりに "hist" を指定
    "device": "cuda",       # GPUを使用するための設定
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.1,
}

# トレーニングの実行
print("GPUでトレーニング中...")
bst = xgb.train(params, dtrain, num_boost_round=100)
print("GPUを使用したトレーニングが完了しました！")
