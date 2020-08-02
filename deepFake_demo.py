from utils import DeepFake

video = 'sample/funny.mp4'
image = 'sample/mark.jpg'

df = DeepFake((image, ), video)
df.generateDeepFake('output.mp4')