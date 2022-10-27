from fastapi import File
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from fastapi.responses import ORJSONResponse
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN as Detector
from deepface import DeepFace

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "https://face2bmi.netlify.app/",
    "https://netlify.app",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MTCNN:

    def __init__(self):
        pass

    def crop_image(self, image):
        # detect faces in the image
        detector = Detector()
        img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        data = detector.detect_faces(img)
        biggest = 0

        if data != []:
            for faces in data:
                box = faces['box']
                # calculate the area in the image
                area = box[3] * box[2]
                if area > biggest:
                    biggest = area
                    bbox = box
            bbox[0] = 0 if bbox[0] < 0 else bbox[0]
            bbox[1] = 0 if bbox[1] < 0 else bbox[1]
            img = img[bbox[1]: bbox[1]+bbox[3], bbox[0]: bbox[0] + bbox[2]]
            # convert from bgr to rgb
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return (True, img)
        else:
            return (False, None)

    def gender_prediction(image):
        gender_result = DeepFace.analyze(image, actions=['gender'])
        print(gender_result)
        return gender_result


@app.post("/upload")
async def upload_image(file: bytes = File(...)):
    image = Image.open(io.BytesIO(file))
    mtcnn = MTCNN()
    status, img = mtcnn.crop_image(image=image)
    print(type(img))
    if status:
        print("face detected...")
        img = np.array(image)
        # print(type(image))
        # print(type(img))
        gender_pred = MTCNN.gender_prediction(img)
        # image = Image.fromarray(img)
        test = np.array(image)

        # create new image of desired size and color (blue) for padding
        res = cv2.resize(img, dsize=(256, 256),
                         interpolation=cv2.INTER_CUBIC)

        image = Image.fromarray(img)

        res = np.expand_dims(res, axis=0)
        print(res.shape)

        # prediction

        def create_model():
            IMG_SHAPE = (256, 256, 3)
            base_model = tf.keras.applications.EfficientNetB6(
                weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
            model_inputs = tf.keras.Input(shape=(256, 256, 3))
            x = base_model(model_inputs, training=False)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Flatten()(x)
            # let's add a fully-connected layer
            '''x = tf.keras.layers.Dense(64,activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(64,activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            '''

            # start passing that fully connected block output to all the different model heads
            y1 = tf.keras.layers.Dense(32, activation='relu')(x)
            y1 = tf.keras.layers.Dropout(0.2)(y1)
            y1 = tf.keras.layers.Dense(16, activation='relu')(y1)
            y1 = tf.keras.layers.Dropout(0.2)(y1)

            y2 = tf.keras.layers.Dense(32, activation='relu')(x)
            y2 = tf.keras.layers.Dropout(0.2)(y2)
            y2 = tf.keras.layers.Dense(16, activation='relu')(y2)
            y2 = tf.keras.layers.Dropout(0.2)(y2)

            y3 = tf.keras.layers.Dense(32, activation='sigmoid')(x)
            y3 = tf.keras.layers.Dropout(0.2)(y3)
            y3 = tf.keras.layers.Dense(16, activation='sigmoid')(y3)
            y3 = tf.keras.layers.Dropout(0.2)(y3)

            # Predictions for each task
            y1 = tf.keras.layers.Dense(
                units=3, activation="linear", name='bmi')(y1)
            y2 = tf.keras.layers.Dense(
                units=3, activation="linear", name='age')(y2)
            y3 = tf.keras.layers.Dense(
                units=3, activation="sigmoid", name='sex')(y3)

            model = tf.keras.Model(inputs=model_inputs, outputs=[y1, y2, y3])
            mae = tf.keras.metrics.MeanAbsoluteError()
            mae1 = tf.keras.metrics.MeanAbsoluteError()

            def precision(y_true, y_pred):  # taken from old keras source code
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / \
                    (predicted_positives + K.epsilon())
                return precision

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                          loss={'bmi': 'mean_squared_error',
                                'age': 'mean_squared_error', 'sex': 'hinge'},
                          metrics={'bmi': mae, 'age': mae1, 'sex': precision})
            return model

        model = create_model()
        weights_path = './saved_face2bmi_model.h5'
        # weights_path = './saved_face2bmi_model.h5'
        for layer in model.layers:
            layer.trainable = False
        print("loading weights...")
        print(model.summary())
        model.load_weights(weights_path)
        print("weights Loaded...")
        data = model.predict(res)

        pred_bmi = [max(x, y, z) for x, y, z in data[0]]
        pred_age = [max(x, y, z) for x, y, z in data[1]]
        pred_sex = [max(x, y, z) for x, y, z in data[2]]

        print("bmi predicted: ", pred_bmi)
        print("age predicted: ", pred_age)
        print("sex predicted: ", [1 if x > 0.5 else 0 for x in pred_sex])

        res = {"containFace": True,
               "bmi": pred_bmi,
               "age": pred_age,
               "sex": gender_pred['gender']
               }
        # return result
        return ORJSONResponse(res)
    else:
        res = {"containFace": False}
        print('No facial image was detected')
        return ORJSONResponse(res)
