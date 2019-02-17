import time

import skimage
import os
from MobileNetModelWrapper import MobileNetModelWrapper
from InceptionResNetModelWrapper import InceptionResNetModelWrapper
from demo_v2.CameraCapture import MyVideoCapture
import cv2
import numpy as np

def run(name):
    """ Set up """
    # Init the camera object
    camera = MyVideoCapture(video_source=0)
    # Set up the window to be fullscreen
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Init the model
    model = MobileNetModelWrapper()
    # Thumbnail Offset
    x_offset = 50
    y_offset = 50
    fps = 0

    """ Start Video display and prediction """
    while True:
        # Determine fps
        start = time.time()

        # Get the image form the camera
        image = camera.get_frame()[1]
        class_name, class_id, scores = model.predict(image)
        thumbnail = skimage.io.imread(os.path.abspath('demo_v2')[:-8] + f'/static/thumbnail_images/{class_name}.png')

        image[y_offset:y_offset + thumbnail.shape[0],
              image.shape[1] - thumbnail.shape[1] - x_offset:-x_offset, :] = thumbnail[:, :, :-1]

        cv2.putText(img=image, text=f'FPS: {fps}', org=(x_offset, y_offset), color=(255, 255, 255),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)
        ret, baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)
        cv2.putText(img=image, text=f'{class_name}', org=(int(image.shape[1]/2)-int(ret[0]/2), y_offset),
                    color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, thickness=1)
        cv2.imshow(name, image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Get the end time
        end = time.time()

        # Calculate frames per second
        fps = np.round(1 / (end - start), 2)
        print(f'Estimated frames per second : {fps} Acc {scores}')

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run('Demo')
