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

<table>
    <thead>
        <tr>
            <th>Source image</th>
            <th>Target domain image</th>
            <th>Beta</th>
            <th>Output</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/source1.jpg?raw=true" width=640></td>
            <td rowspan=3><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/target1.jpg?raw=true" width=640></td>
            <td>0.001</td>
            <td><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/output1_0.001.jpg?raw=true" width=640></td>
        </tr>
        <tr>
            <td>0.01</td>
            <td><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/output1_0.01.jpg?raw=true" width=640></td>
        </tr>
        <tr>
            <td>0.1</td>
            <td><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/output1_0.1.jpg?raw=true" width=640></td>
        </tr>
        <tr>
            <td rowspan=3><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/source2.jpg?raw=true" width=640></td>
            <td rowspan=3><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/target2.jpg?raw=true" width=640></td>
            <td>0.001</td>
            <td><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/output2_0.001.jpg?raw=true" width=640></td>
        </tr>
        <tr>
            <td>0.01</td>
            <td><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/output2_0.01.jpg?raw=true" width=640></td>
        </tr>
        <tr>
            <td>0.1</td>
            <td><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/output2_0.1.jpg?raw=true" width=640></td>
        </tr>
        <tr>
            <td rowspan=3><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/source3.jpg?raw=true" width=640></td>
            <td rowspan=3><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/target3.jpg?raw=true" width=640></td>
            <td>0.001</td>
            <td><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/output3_0.001.jpg?raw=true" width=640></td>
        </tr>
        <tr>
            <td>0.01</td>
            <td><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/output3_0.01.jpg?raw=true" width=640></td>
        </tr>
        <tr>
            <td>0.1</td>
            <td><img src="https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/images/output3_0.1.jpg?raw=true" width=640></td>
        </tr>
    </tbody>
</table>


Run unit tests
--------------
```
$ python3 tests/test_fourier.py
```

License
-------

This repository is shared under an [MIT](https://github.com/luiscarlosgph/fourier-domain-adaptation/blob/main/LICENSE) license.


Author
------

Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com), 2020-2022.


