from utils import DeepFake

video = 'sample/funny.mp4'
image = 'sample/mark.jpg'

DeepFake((image, ), video).generateDeepFake('output.mp4')