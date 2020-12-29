import cv2
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from PIL import Image
import time

#telloVideo = cv2.VideoCapture("udp://@0.0.0.0:11111")
telloVideo = cv2.VideoCapture("/Users/nitinkak/git_repos/course-v3/nbs/dl1/data/source/me.mp4")
# wait for frame
ret = False
# scale down
scale = 3
bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
))


def segment_body_part(frame, part_names):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_array = tf.keras.preprocessing.image.img_to_array(pil_img)
    result = bodypix_model.predict_single(image_array)
    mask = result.get_mask(threshold=0.92)
    colored_mask = result.get_colored_part_mask(mask, part_names=part_names)
    return colored_mask


def preprocess_frame(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dsize=dim)


i = 0
start_time = time.time()
while True:
    # Capture frame-by-framestreamon
    ret, frame = telloVideo.read()
    if ret:
        # Display the resulting frame
        frame = preprocess_frame(frame)
        if i % 25 in range(26):
            cv2.imshow('DJI Tello',
                       segment_body_part(frame, ['left_face', 'right_face']).astype(
                           'uint16') * 255)
        else:
            cv2.imshow('DJI Tello', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("test.jpg", frame)  # writes image test.bmp to disk
        print("Take Picture")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1
    if i % 100 == 0:
        end_time = time.time()
        print('Response time(100 frames): {avg_time}'.format(avg_time=(end_time - start_time) / 100))
        start_time = end_time

# When everything done, release the capture
telloVideo.release()
cv2.destroyAllWindows()
