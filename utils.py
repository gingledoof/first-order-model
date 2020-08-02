import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
from demo import load_checkpoints
from demo import make_animation
from skimage import img_as_ubyte
from moviepy.editor import *
import cv2
import ffmpeg
from tqdm import tqdm
import os

class DeepFake:

    def getImage(self, images):
        imgs = []
        for image in images:
            imgs.append(resize(imageio.imread(image), (256, 256))[..., :3])
        return imgs

    def getVideo(self, video):
        driving_video = []

        reader = imageio.get_reader(video)
        try:
            if self.faceRec:
                max_faces = 0
                for im in reader:

                    faces = self.facialRecog(im)

                    if max_faces != len(faces):
                        for _ in range(len(faces)):
                            driving_video.append([])
                        max_faces = len(faces)

                    for i, (x, y, w, h) in enumerate(faces):
                        crop = resize(im[y:y+h, x:x+w], (256, 256))
                        driving_video[i].append(crop)
            else:
                faces = []
                for im in reader:
                    faces.append(resize(im, (256, 256)))

                driving_video.append(faces)

        except RuntimeError:
            pass

        return driving_video

    def __init__(self, image, video, faceRec = False):
        self.faceRec = faceRec
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        warnings.filterwarnings("ignore")
        self.source_images = self.getImage(image)
        self.driving_video = self.getVideo(video)
        v = VideoFileClip(video)
        self.video_fps = v.fps
        self.audio = v.audio
        self.predictions = []


    def generateDeepFake(self, output_dir, relative=True, adapt_movement_scale=True):

        generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                                                  checkpoint_path='vox-cpk.pth.tar')

        for i, faces in enumerate(self.driving_video):



            self.predictions.append(make_animation(self.source_images[i],
                                         faces,
                                         generator,
                                         kp_detector,
                                         relative= relative,
                                         adapt_movement_scale=adapt_movement_scale)
                               )

            self.saveVideo(self.predictions[i], str(i)+output_dir)


    def saveVideo(self, predictions, output_dir):
        v = cv2.VideoWriter("temp.mp4", 0, self.video_fps, (256, 256))

        for frame in predictions:
            frame = cv2.cvtColor(img_as_ubyte(frame), cv2.COLOR_RGB2BGR)
            v.write(frame)
        v.release()

        v = VideoFileClip("temp.mp4")
        v = v.speedx( v.duration / self.audio.duration )
        v = v.set_audio(self.audio)
        v.write_videofile(output_dir)
        os.remove("temp.mp4")
        print("Success! " + str(output_dir))



    def display(source, driving, generated=None):
        fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

        ims = []
        for i in range(len(driving)):
            cols = [source]
            cols.append(driving[i])
            if generated is not None:
                cols.append(generated[i])
            im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
            plt.axis('off')
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
        plt.close()
        return ani

    def facialRecog(self, image, scale=1.2):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        return self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=5,
            minSize=(30, 30)
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )