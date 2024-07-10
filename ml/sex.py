import os
import numpy as np
from PIL import Image
from pathlib import Path
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.simplefilter('ignore')

model_dir   = "data/model_b.h5"

resize_settings = (64, 64)

def sex(path, model=None, augment_times=10):
    
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # 正規化

    model = load_model(model_dir)

    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_images = datagen.flow(img_array, batch_size=1)

    predictions = []
    for _ in range(augment_times):
        aug_img = next(augmented_images)[0]
        aug_img = np.expand_dims(aug_img, axis=0)
        prediction = model.predict(aug_img, verbose=0)[0][0]
        predictions.append(prediction)

    avg_prediction = np.mean(predictions)
    avg_female_prob = 1- avg_prediction

    # 推論の実行
    predictions = model.predict(img_array, verbose=0)
    prediction = predictions[0][0]
    female_prob = predictions[0][0]
    male_prob = 1 - female_prob

    # 結果の表示
    print(path)
    print(f"male probability: {female_prob * 100:.2f}%")
    print(f"Female probability: {male_prob * 100:.2f}%")

    result = {
        "メス": f"{male_prob * 100:.2f}%",
        "オス": f"{female_prob * 100:.2f}%",
        "メス(avg)": f"{avg_female_prob * 100:.2f}%",
        "オス(avg)": f"{avg_prediction * 100:.2f}%"
    }
    return result


if __name__ == "__main__":
    
    labels = ["メス", "オス"]
    model = load_model(model_dir)
    

    directory = "/Users/kota-izk/Documents/worksplace/Python/centipedeAI/male_3/"
    os.system(f"rm {directory}.DS_Store")
    for f in Path(directory).rglob('*.jpg'):
        print(f)
        print(sex(str(f)))


