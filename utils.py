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
import face_recognition as fr

class DeepFake:

    def getImage(self, images):
        imgs = []
        for image in images:
            imgs.append(resize(imageio.imread(image), (256, 256))[..., :3])
        return imgs

    def getVideo(self, video):

        return self.defaultVideo(video)
        #if self.static_face:
        #  return self.singleFaceVideo(video)
        #else:
        #    return self.multiFaceVideo(video)

    def defaultVideo(self, video):
        driving_video = []
        reader = imageio.get_reader(video)

        try:
            faces = []
            for im in reader:
                faces.append(resize(im, (256, 256)))
        except:
            RuntimeError

        driving_video.append(faces)
        return driving_video

    def singleFaceVideo(self, video):

        driving_video = []
        reader = imageio.get_reader(video)

        try:
            if self.static_face:
                driving_video.append([])
                init = True
                (x, y, w, h) = (None, None, None, None)
                for im in reader:
                    if init:
                        (x, y, w, h) = self.facialRecog(im)[0]

                    im = im[y:y + h, x:x + w]
                    crop = resize(im, (256, 256))
                    driving_video[0].append(crop)
                    init = False


            elif self.faceRec:
                for im in reader:

                    faces = self.facialRecog(im)

                    (x, y, w, h) = faces[0]

                    if self.faceRec:
                        if not self.coordet.grad(x, y, w, h):
                            (x, y, w, h) = (self.coordet.getXAvg(),
                                            self.coordet.getYAvg(),
                                            self.coordet.getWAvg(),
                                            self.coordet.getHAvg())

                    im = im[y:y + h, x:x + w]
                    crop = resize(im, (256, 256))
                    driving_video[i].append(crop)

            else:
                faces = []
                for im in reader:
                    faces.append(resize(im, (256, 256)))

                driving_video.append(faces)

        except RuntimeError:
            pass

        self.saveDev(driving_video[0])
        return driving_video

    def multiFaceVideo(self, video):
        driving_video = []
        reader = imageio.get_reader(video)

        try:
            if self.static_face:
                face_coords = self.facialRecog(reader[0])

                for coord in face_coords:
                    driving_video.append([])

                for im in reader:
                    for coord in face_coords:
                        (x, y, w, h) = coord
                        im = im[y:y + h, x:x + w]
                        crop = resize(im, (256, 256))
                        driving_video[i].append(crop)

            elif self.faceRec:
                print("Not implemented yet!")
                raise RuntimeError

            else:
                print("Cannot do multi face detection without facial recognition")
                raise RuntimeError




        except:
            RuntimeError

    def generateDeepFake(self, output_dir, relative=True, adapt_movement_scale=True, adv=False, cpu=False):

        cp = 'vox-cpk.pth.tar'
        if adv:
            cp = 'vox-adv-cpk.pth.tar'

        generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                                                  checkpoint_path=cp,
                                                  cpu=cpu)

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
        v.close()
        os.remove("temp.mp4")
        print("Success! " + str(output_dir))

    def saveDev(self, arr):
        v = cv2.VideoWriter("dev.mp4", 0, self.video_fps, (256, 256))

        for frame in arr:
            frame = cv2.cvtColor(img_as_ubyte(frame), cv2.COLOR_RGB2BGR)
            v.write(frame)
        v.release()

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

    def facialRecog(self, image):
        # Detect faces in the image
        return fr.face_locations(image)

    def __init__(self, image, video, thres = 30, faceRec = False, static_face = True):
        self.coordet = CoordDet(thres)
        v = VideoFileClip(video)
        self.video_fps = v.fps
        self.audio = v.audio
        self.predictions = []
        self.faceRec = faceRec
        self.static_face = static_face
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        warnings.filterwarnings("ignore")
        self.source_images = self.getImage(image)
        self.driving_video = self.getVideo(video)
        print('Image and Video Loaded')


class CoordDet:

    def __init__(self, thres):
        self.x_sum = 0
        self.y_sum = 0
        self.w_sum = 0
        self.h_sum = 0
        self.thres = thres
        self.iter = 0

    def update(self, x, y, w, h):
        self.x_sum += x
        self.y_sum += y
        self.h_sum += h
        self.w_sum += w
        self.iter += 1

    def grad(self, x, y, w, h):
        self.update(x, y, w, h)

        if ((abs(self.x_sum / self.iter - x)) > self.thres) or ((abs(self.y_sum / self.iter - y)) > self.thres):
            return True

    def getXAvg(self):
        return self.x_sum / self.iter

    def getYAvg(self):
        return self.y_sum / self.iter

    def getWAvg(self):
        return self.w_sum / self.iter

    def getHAvg(self):
        return self.h_sum / self.iter