import os
import pandas as pd
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

################################################################################ 학습용 데이터셋 준비

train_csv = './DeepPCB_split/trainval_label.csv'
df = pd.read_csv(train_csv)
train_image_dir = './DeepPCB_split/train'

#filepath -> csv 파일의 filename 열에 저장되어 있는 이미지 이름과 동일한 이미지의 실제 경로(폴더)를 filepath에 저장
df['filepath'] = df['filename'].apply(lambda x: os.path.join(train_image_dir, x))
'''
| filename  | filepath                  |
| --------- | ------------------------- |
| file1.jpg | ./train\_images/file1.jpg |
| file2.jpg | ./train\_images/file2.jpg |
| file3.jpg | ./train\_images/file3.jpg |
'''

#labels: 정상(0), 불량(1) 이미지를 판단하기 위한 라벨링은 csv 파일의 두번째 열인 Defects 이용
#model 학습에 사용하기 위해 각각의 열을 list화
filepaths = df['filepath'].tolist()
labels = df['Defects'].tolist()

IMG_SIZE = 224  # CNN에서 보통통 사용하는 표준 크기

def decode_img(img_path, label):
    # 이미지 읽기
    img = tf.io.read_file(img_path)
    # JPEG 디코딩, 채널 3개 (RGB)
    img = tf.image.decode_jpeg(img, channels=3)
    # 이미지 크기 조정
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    # 픽셀값 0~1 정규화
    img = img / 255.0
    return img, label

BATCH_SIZE = 32

dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
dataset = dataset.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

################################################################################## 테스트용 데이터셋 준비
test_csv_path = './DeepPCB_split/test_label.csv'  # 예시
test_df = pd.read_csv(test_csv_path)
test_dir = './DeepPCB_split/test'

test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(test_dir, x))

test_filepaths = test_df['filepath'].tolist()
test_labels = test_df['Defects'].tolist()

def make_dataset(files, labels, shuffle=False, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

test_dataset = make_dataset(test_filepaths, test_labels, shuffle=False, batch_size=32)

################################################################################# CNN 모델 구축

#리스트 분할(80% 학습, 20% 검증)
train_files, val_files, train_labels, val_labels = train_test_split(
    filepaths, labels, test_size=0.2, random_state=42, stratify=labels
)

# 2. tf.data.Dataset 만들기
train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_dataset = train_dataset.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
val_dataset = val_dataset.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

IMG_SIZE = 224  # 이미지 크기
NUM_CLASSES = 2  # 정상(0), 불량(1)

#모델 정의(1~5번)
#model_1 -> 보통의 CNN, 4번의 Conv2D + MaxPooling으로 이루어져 있음음
model_1 = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
model_1.summary()
model_1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#모델 학습
history_1 = model_1.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=10)

#model_2 -> model_1에 학습 성능을 안정화시켜주는 Batch Normalization 레이어층을 추가
model_2 = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
model_2.summary()
model_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#모델 학습습
history_2 = model_2.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=10)

#model_3 -> 죽은 뉴런 문제를 방지할 수 있는 LeakyReLU를 사용한 모델
model_3 = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D(),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128),
    layers.LeakyReLU(alpha = 0.1),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
model_3.summary()
model_3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#모델 학습습
history_3 = model_3.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=10)
#학습 결과 저장
def save_model_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history.history, f)

#model 1 ~ 5번에 대한 학습 결과를 Model results 폴더에 저장(시각화 하기 위함)
save_model_history(history_1, "./DeepPCB_split/Model results/model1_history.json")
save_model_history(history_2, "./DeepPCB_split/Model results/model2_history.json")
save_model_history(history_3, "./DeepPCB_split/Model results/model3_history.json")
#평가
test_loss1, test_acc1 = model_1.evaluate(test_dataset)
test_loss2, test_acc2 = model_2.evaluate(test_dataset)
test_loss3, test_acc3 = model_3.evaluate(test_dataset)
#평가 결과 출력
test_accs = [test_acc1, test_acc2, test_acc3]
model_names = ["Model 1", "Model 2", "Model 3"]

plt.bar(model_names, test_accs)
plt.title("Test Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.show()
