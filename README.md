# <center>Детектирование медицинских масок</center>

<img src=./notebook/imgs/readme_example.png width=1200 align=center></img>

Датасет для обучения был взят с [<i><b>Kaggle</b></i>](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

Было обучено 2 модели с высоким показателем $\text{mAP@50}$:
* [<i><b>RetinaNet</b></i>](./notebook/retinanet.ipynb) (ссылка на ноутбук)
* [<i><b>YOLOv8</b></i>](./notebook/yolo.ipynb) (ссылка на ноутбук)

|Модель|$\text{mAP@50}$|
|-|-|
|<i><b>RetinaNet</b></i>|$0.98$|
|<i><b>YOLOv8</b></i>|$0.96$|