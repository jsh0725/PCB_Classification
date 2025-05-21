import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

############################################################################# 학습용 데이터셋 준비

csv_path = './DeepPCB_split/trainval_label.csv'
df = pd.read_csv(csv_path)

# Defects 컬럼이 0이면 정상(0), 1 이상이면 불량(1)
df['label'] = df['Defects'].apply(lambda x: 0 if x == 0 else 1)

image_dir = './DeepPCB_split/train'

df['filepath'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))

filepaths = df['filepath'].tolist()
labels = df['label'].tolist()

IMG_SIZE = 224  # 원하는 이미지 크기

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
# test CSV 경로 (trainval_label.csv와 다르다면 따로 지정)
test_csv_path = './DeepPCB_split/test_label.csv'  # 예시

# CSV 불러오기
test_df = pd.read_csv(test_csv_path)

# 라벨링 (train과 동일하게)
test_df['label'] = test_df['Defects'].apply(lambda x: 0 if x == 0 else 1)

# 이미지 경로 생성
test_dir = './DeepPCB_split/test'
test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(test_dir, x))

test_filepaths = test_df['filepath'].tolist()
test_labels = test_df['label'].tolist()

def make_dataset(files, labels, shuffle=False, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

test_dataset = make_dataset(test_filepaths, test_labels, shuffle=False, batch_size=32)

#################################################################################CNN 모델 정의 
# 예를 들어 80% 학습, 20% 검증 분할
split_index = int(len(filepaths) * 0.8)

train_dataset = dataset.take(split_index)
val_dataset = dataset.skip(split_index)

IMG_SIZE = 224  # 이미지 크기
NUM_CLASSES = 2  # 정상(0), 불량(1)

#모델 학습
model = models.Sequential([
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

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 10

history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS)

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")