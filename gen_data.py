import cv2, h5py, sys


if len(sys.argv) < 2:
    print('Usage: python gen_data.py <input_file> <output_file>.')
    exit(-1)

import keras
from keras_retinanet.models.resnet import custom_objects

model = keras.models.load_model('./retinanet.h5', custom_objects=custom_objects)
vid = cv2.VideoCapture(sys.argv[0])

with h5py.File(sys.argv[1]) as file:
    while True:
        ok, frame = vid.read()
        if not ok:
            break

        _, _, detections = model.predict_on_batch(np.expand_dims(frame, axis=0))

        predicted_labels = argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

        good = []
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < 0.5:
                continue
            b = detections[0, idx, :4].astype(int)
            good.append(b)

        file.create_dataset('frame' + str(counter), data=good)
        file.flush()
        counter += 1
