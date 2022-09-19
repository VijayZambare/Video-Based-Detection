# deepface
<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="200" height="240"></p>

Deepface is a lightweight [face recognition](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and facial attribute analysis ([age](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [gender](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [emotion](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) and [race](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/)) framework for python. It is a hybrid face recognition framework wrapping **state-of-the-art** models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/), [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [`ArcFace`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/), [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/) and `SFace`.

Experiments show that human beings have 97.53% accuracy on facial recognition tasks whereas those models already reached and passed that accuracy level.

## Installation [![PyPI](https://img.shields.io/pypi/v/deepface.svg)](https://pypi.org/project/deepface/) [![Conda](https://img.shields.io/conda/vn/conda-forge/deepface.svg)](https://anaconda.org/conda-forge/deepface)

The easiest way to install deepface is to download it from [`PyPI`](https://pypi.org/project/deepface/). It's going to install the library itself and its prerequisites as well.

```shell
$ pip install deepface
```

DeepFace is also available at [`Conda`](https://anaconda.org/conda-forge/deepface). You can alternatively install the package via conda.

```shell
$ conda install -c conda-forge deepface
```

Then you will be able to import the library and use its functionalities.

```python
from deepface import DeepFace
```


```python
backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]

#face verification
obj = DeepFace.verify(img1_path = "img1.jpg", 
        img2_path = "img2.jpg", 
        detector_backend = backends[4]
)

#face recognition
df = DeepFace.find(img_path = "img.jpg", 
        db_path = "my_db", 
        detector_backend = backends[4]
)

#embeddings
embedding = DeepFace.represent(img_path = "img.jpg", 
        detector_backend = backends[4]
)

#facial analysis
demography = DeepFace.analyze(img_path = "img4.jpg", 
        detector_backend = backends[4]
)

#face detection and alignment
face = DeepFace.detectFace(img_path = "img.jpg", 
        target_size = (224, 224), 
        detector_backend = backends[4]
)
```

Face recognition models are actually CNN models and they expect standard sized inputs. So, resizing is required before representation. To avoid deformation, deepface adds black padding pixels according to the target size argument after detection and alignment.

