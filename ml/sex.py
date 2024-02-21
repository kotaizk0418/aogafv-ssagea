import os
import numpy as np
from PIL import Image
from pathlib import Path
from keras.models import load_model
import warnings
warnings.simplefilter('ignore')

model_dir   = "data/cnn_sexDataV3ModelV12"

resize_settings = (64, 64)

def sex(path, model=None):
    
    labels = ["メス", "オス"]
    if not model:
        model = load_model(model_dir)
    X     = []                               # 推論データ格納
    image = Image.open(path)                 # 画像読み込み
    image = image.convert("RGB")             # RGB変換
    image = image.resize(resize_settings)    # リサイズ
    data  = np.asarray(image)                # 数値の配列変換
    data  = data / 255.0
    X.append(data)
    X     = np.array(X)
    
    # モデル呼び出し
    
    
    # numpy形式のデータXを与えて予測値を得る
    model_output = model.predict([X])[0]
    # 推定値 argmax()を指定しmodel_outputの配列にある推定値が一番高いインデックスを渡す
    predicted = model_output.argmax()
    result = {}
    print("PATH: ", path)
    for index, modelo in enumerate(model_output):
        # アウトプット正答率
        accuracy = round(float(modelo * 100), 2)

        print("{0}: {1}%".format(labels[index], accuracy))
        result["{}".format(labels[index])] = "{}%".format(accuracy)

    return result


if __name__ == "__main__":
    
    labels = ["メス", "オス"]
    model = load_model(model_dir)
    

    directory = "/Users/kota-izk/Documents/worksplace/Python/centipedeAI/male_2/"
    os.system(f"rm {directory}.DS_Store")
    for f in Path(directory).rglob('*'):
        path = directory + f.stem + '.jpg'
        image = Image.open(path)
        print(sex(path))


