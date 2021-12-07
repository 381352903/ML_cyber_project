# ML_cyber_project
The project is to design a backdoor detector for BadNets trained on the YouTube Face dataset.

**Authors:** Tairun Meng, Chenyan Zhou, Xinzhu Han

## I. Data Arrangement

Replace the empty "data" folder with the data folder in the Google Drive. The data folder can be downloaded here: https://drive.google.com/drive/folders/1GbEHNISuPIO-QEfJyOChjGIrNwFTAF_G?usp=sharing

## II. Run the tests

The Python files in the GoodNets folder are used to evaluate each GoodNet. 
To run the test, run ``` python3 <good_net_to_evaluate.py> <test_image.png> ```.
The output class is in the range of [0, 1283]. The GoodNet will output 1283 if the test image is poisoned, else the output class is in rage[0, 1282].

Example:
```bash
python3 GoodNets/evaluate_anonymous_1.py test_images/anonymous_1_poisoned_1.png
python3 GoodNets/evaluate_sunglasses.py test_images/sunglass_poisoned_2.png 
```

## II.Dependencies

1. numpy<br>
2. cv2<br>
3. scipy<br>
4. h5py<br>
5. sys <br>
6. keras <br>
7. scipy.stats
