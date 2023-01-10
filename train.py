from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint


train_gen = ImageDataGenerator(horizontal_flip = True)
train_data_gen = train_gen.flow_from_directory("Data/seg_train/",
                                              target_size = (150,150),
                                              color_mode = 'rgb',
                                              batch_size =32,
                                              shuffle=True,
                                              class_mode = 'categorical')

test_gen = ImageDataGenerator()

test_data_gen = train_gen.flow_from_directory("Data/seg_test/",
                                              target_size = (150,150),
                                              color_mode = 'rgb',
                                              batch_size =32,
                                              shuffle=False,
                                              class_mode = 'categorical')


first = tf.keras.Sequential([
    tf.keras.layers.Input((150,150,3)),
    tf.keras.layers.Lambda(preprocess_input)
])

mobile = MobileNetV2(include_top=False,input_shape=(150,150,3))

last = tf.keras.Sequential([
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation='softmax')
])

model = tf.keras.Sequential([first, mobile, last])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),loss='categorical_crossentropy',
             metrics=['accuracy'])

model.build(((None, 150,150,3)))


model_chk = ModelCheckpoint('model/',save_best_only=True)
history = model.fit(train_data_gen, validation_data=test_data_gen, epochs=10, callbacks=[model_chk])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('intel-classifier.tflite', 'wb') as f:
    f.write(tflite_model)