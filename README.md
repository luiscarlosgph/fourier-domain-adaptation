Fourier Domain Adaption (FDA)
-----------------------------

This Python package performs a classic frequency domain adaptation.

For more info, check this paper by Yanchao Yang and Stefano Soatto in CVPR 2020: 

[**FDA: Fourier Domain Adaptation for Semantic Segmentation**](https://arxiv.org/abs/2004.05498)


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

```
import fda

# Read source and target images
source_im = cv2.imread('source.jpg')
target_im = cv2.imread('target.jpg')

# Perform domain adaptation
adapted_im = fda.fda(source,_im, target_im, beta=0.005)
```

Run domain adaptation
---------------------
```
$ python3 -m 
```

Some examples of the domain adaptation
--------------------------------------

| Source      | Target      | Result |
| ----------- | ----------- | ------ |
| X           | Y           | Z

License
-------

MIT


Author
------

Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com), 2020-2022.


