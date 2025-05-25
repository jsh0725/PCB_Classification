import os
import pandas as pd
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import random
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- GPU 메모리 할당 방식 설정 (TensorFlow 임포트 후 바로) ---
# 메모리 조각화 문제를 완화하고 GPU 메모리 할당 효율성을 높입니다.
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' 
# ---

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 증가 설정을 활성화하여 TensorFlow가 필요한 만큼만 GPU 메모리를 할당하도록 합니다.
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

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

IMG_SIZE = 224  # CNN에서 보통 사용하는 표준 크기
BATCH_SIZE = 32

#decode_img: 이미지 경로 -> 실제 이미지 tensor로 변환하는 전처리 함수
#위 함수를 tf.data.Dataset에 연결해서 학습에 쓸 최종 데이터를 만든다.
def decode_img(img_path, label):  
    # 이미지 경로로읽기
    img = tf.io.read_file(img_path)
    # JPEG 디코딩, 흑백 이미지이므로 channels = 1
    img = tf.image.decode_jpeg(img, channels=1)
    #명시적 형 변환(float 32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 이미지 크기 재지정정
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label


#리스트 분할(80% 학습, 20% 검증)
train_files, val_files, train_labels, val_labels = train_test_split(
    filepaths, labels, test_size=0.2, random_state=42, stratify=labels
)

#Dataset 만들기
#학습용 Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_dataset = train_dataset.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#검증용 Dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
val_dataset = val_dataset.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

################################################################################## 테스트용 데이터셋 준비
test_csv_path = './DeepPCB_split/test_label.csv'  # 예시
test_df = pd.read_csv(test_csv_path)
test_dir = './DeepPCB_split/test'

test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(test_dir, x))

test_filepaths = test_df['filepath'].tolist()
test_labels = test_df['Defects'].tolist()

test_dataset = tf.data.Dataset.from_tensor_slices((test_filepaths, test_labels))
test_dataset = test_dataset.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

################################################################################# CNN 모델 구축

EPOCH = 20
IMG_SIZE = 224  # 이미지 크기

#모델 정의(1~5번)
#binary classification은 이진 분류, 즉 단 하나의 선택만 필요하므로 softmax 보다 sigmoid가 더 적절하다.
#또한 손실함수도 sparse_categorical_crossentropy보다는 binary_crossentropy가 더 적절하다.
#model_1은 softmax + sparse_categorical_crossentropy, model_2은 softmax + binary_crossentropy로 구성하여 비교해보자.
#출력 노드(Dense()) 또한 각 클래스에 대한 확률을 예측하기 보단(노드가 2개이기 보단) 하나의 확률값에 대해 정상 / 불량을 판단하는게 더 효율적이다.
# 이 또한 포함하여 코드를 구성해 비교

#model_1 : softmax + sparse_categorical_crossentropy, 출력층 2개
model_1 = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    
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
    layers.Dense(2, activation='softmax')
])
model_1.summary()
model_1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#모델 학습
history_1 = model_1.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs = EPOCH)

#model_2 : sigmoid + binary_crossentropy, 출력층 1개
model_2 = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    
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
    layers.Dense(1, activation='sigmoid')
])
model_2.summary()
model_2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#모델 학습
history_2 = model_2.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs = EPOCH)

#model_3 : 기존 Model_2에 향상된 성능을 위해 BatchNormalization layer을 추가, 안정적이면서 부드러운 학습
model_3 = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 1)),

    layers.Conv2D(16, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid') 
])
model_3.summary()
model_3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy'])
#모델 학습
history_3 = model_3.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs = EPOCH)

#model_4 -> residual connections를 추가하여 정보 손실을 방지하는 모델
def residual_block(x, filters):
    shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)  # 크기 맞추기 

    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])  # 동일한 크기로 유지 
    x = layers.ReLU()(x)

    return x

# Model 4: ResNet 구조
input_layer = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = layers.Conv2D(64, (3,3), activation='relu', padding="same")(input_layer)
x = layers.BatchNormalization()(x)

x = residual_block(x, 64)
x = layers.MaxPooling2D()(x)

x = residual_block(x, 128)
x = layers.MaxPooling2D()(x)

x = residual_block(x, 256)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output_layer = layers.Dense(2, activation='softmax')(x)

model_4 = models.Model(inputs=input_layer, outputs=output_layer)
model_4.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# 모델 학습
model_4.summary()
history_4 = model_4.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=EPOCH)

# model_5 -> dense connection 구조를 추가하여 정보 손실을 방지하는 모델
def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        conv_layer = layers.Conv2D(growth_rate, (3, 3), padding='same', activation='relu')(x)
        conv_layer = layers.BatchNormalization()(conv_layer)
        x = layers.Concatenate()([x, conv_layer])  # Dense Connection 

    return x 

# Model 5: DenseNet 구조
input_layer = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = layers.Conv2D(16, (3,3), activation='relu', padding="same")(input_layer)
x = layers.BatchNormalization()(x)

x = dense_block(x, 2, 16)
x = layers.MaxPooling2D()(x)

x = dense_block(x, 2, 32)
x = layers.MaxPooling2D()(x)

x = dense_block(x, 2, 64)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output_layer = layers.Dense(2, activation='softmax')(x)

model_5 = models.Model(inputs=input_layer, outputs=output_layer)
model_5.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# 모델 학습
model_5.summary()
history_5 = model_5.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=EPOCH)


#학습 결과 저장
def save_model_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history.history, f)

#model 1 ~ 5번에 대한 학습 결과를 Model results 폴더에 저장(시각화 하기 위함)
save_model_history(history_1, "./DeepPCB_split/Model results/model1_history.json")
save_model_history(history_2, "./DeepPCB_split/Model results/model2_history.json")
save_model_history(history_3, "./DeepPCB_split/Model results/model3_history.json")
save_model_history(history_4, "./DeepPCB_split/Model results/model4_history.json")
save_model_history(history_5, "./DeepPCB_split/Model results/model5_history.json")
#평가
test_loss1, test_acc1 = model_1.evaluate(test_dataset)
test_loss2, test_acc2 = model_2.evaluate(test_dataset)
test_loss3, test_acc3 = model_3.evaluate(test_dataset)
test_loss4, test_acc4 = model_4.evaluate(test_dataset)
test_loss5, test_acc5 = model_5.evaluate(test_dataset)

test_accs = [test_acc1, test_acc2, test_acc3, test_acc4, test_acc5]
model_names = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]

#평가 결과 시각화화(Graph)
for i, acc in enumerate(test_accs):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center')

plt.bar(model_names, test_accs)
plt.title("Test Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.show()