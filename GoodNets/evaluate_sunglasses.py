import keras
import sys
import h5py
import numpy as np
import cv2
import scipy
import scipy.stats
import warnings

# sunglasses_poisoned_data.h5
filename = str(sys.argv[1]) 
model_filename = "models/sunglasses_bd_net.h5"
processced_image_clean = "data/clean_validation_data.h5"

def get_data(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label']) 
    x_data = x_data.transpose((0,2,3,1)) 
    return x_data, y_data

def preprocess_data(x_data):
    return x_data/255

def process_image(background, overlay):
    processced_image = \
        cv2.addWeighted(background,1,overlay,1,0, dtype=cv2.CV_64F)
    return processced_image

# Calcuate Entropy
def calculate_entropy(model, background_img, x_validation, n):
    x_perturbed = [0] * n
    idx = np.random.randint(x_validation.shape[0], size=n)
    for i in range(n) :
        x_perturbed[i] = \
            process_image(background_img, x_validation[idx[i]])
    prediction = model.predict(np.array(x_perturbed))
    return -np.nansum(prediction * np.log2(prediction))

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    x_test = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    x_test = cv2.cvtColor(x_test, cv2.COLOR_BGR2RGB)
    x_test = x_test[np.newaxis,:] # 1x55x47x3
    x_test = preprocess_data(x_test) 
    x_clean_validation, y_clean_test =  get_data(processced_image_clean)
    x_clean_validation_test = preprocess_data(x_clean_validation)

    model = keras.models.load_model(model_filename)
    ## GoodNet Start ##
    input_res = model.predict(x_test)
    label = input_res[0].shape[0]  
    n_img = 25

    # Processe images
    x_test_num = x_test.shape[0]
    entropy = [0] * x_test_num

    # Calculate entropy
    for i in range(x_test_num):
        entropy[i] = calculate_entropy(model, x_test[i], x_clean_validation_test, n_img)

    entropy = [x / n_img for x in entropy] 
    threshold = 0.22104038464605122

    # Predict
    y_predict = np.argmax(input_res, axis = 1)
    for i in range(x_test_num):
        if entropy[i] < threshold:
            y_predict[i] = label

    print(y_predict[0])

if __name__ == '__main__':
    main()
