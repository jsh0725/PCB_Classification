import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

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
#80% 학습, 20% 검증 분할
split_index = int(len(filepaths) * 0.8)

train_dataset = dataset.take(split_index)
val_dataset = dataset.skip(split_index)

IMG_SIZE = 224  # 이미지 크기
NUM_CLASSES = 2  # 정상(0), 불량(1)

#모델 정의
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

#모델 학습
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=15)
#평가
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")