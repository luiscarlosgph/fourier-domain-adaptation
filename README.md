Fourier Domain Adaption (FDA)
-----------------------------

This Python package implements a classic frequency domain adaptation, as shown in: 

 [**FDA: Fourier Domain Adaptation for Semantic Segmentation, Yanchao Yang and Stefano Soatto, CVPR 2020**](https://arxiv.org/abs/2004.05498)


Install with pip
----------------

```
$ python3 -m pip install fda --user
```


Install from source
-------------------

```
$ python3 setup.py install --user
```


Exemplary code snippet
----------------------

```python
import fda

# Read source and target images
source_im = cv2.imread('source.jpg')
target_im = cv2.imread('target.jpg')

# Perform domain adaptation
adapted_im = fda.fda(source,_im, target_im, beta=0.005)
```


Run domain adaptation on a single image
---------------------------------------

```
$ python3 -m fda.run --source source.jpg --target target.jpg --output output.jpg --beta 0.005
```


Some examples of the domain adaptation
--------------------------------------

| Source image | Target domain image | Result |
| ------------ | ------------------- | ------ |
| <img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/source1.jpg?raw=true" width=200> | <img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/target1.jpg?raw=true" width=200>           | <img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/output1.jpg?raw=true" width=200>

License
-------

MIT


Author
------

Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com), 2020-2022.


