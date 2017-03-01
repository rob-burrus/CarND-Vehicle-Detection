#Train SVC
import os
import glob
import time
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_utils import extract_features

def load_images():
    basedir = 'vehicles/'
    image_types = os.listdir(basedir)
    cars = []
    for imtype in image_types:
        cars.extend(glob.glob(basedir+imtype+'/*'))
    print ('Number of Vehicle Images found: ', len(cars))
    with open('cars.txt', 'w') as f:
        for fn in cars:
            f.write(fn + '\n')

    basedir = 'non-vehicles/'
    image_types = os.listdir(basedir)
    notcars = []
    for imtype in image_types:
        notcars.extend(glob.glob(basedir+imtype+'/*'))
    print ('Number of Non-Vehicle Images found: ', len(notcars))
    with open('non-cars.txt', 'w') as f:
        for fn in notcars:
            f.write(fn + '\n')

    return cars, notcars


def train_svc(cars, notcars):
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    t = time.time()
    #n_samples=1000
    #random_idxs = np.random.randint(0, len(cars), n_samples)
    test_cars = cars#np.array(cars)[random_idxs]
    test_notcars = notcars#np.array(notcars)[random_idxs]


    car_features = extract_features(test_cars, cspace=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(test_notcars, cspace=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    print(time.time()-t, 'Seconds to compute features...')

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    X_scaler = StandardScaler().fit(X)

    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0,100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    dist_pickle = {}
    dist_pickle["svc"] = svc
    dist_pickle["X_scaler"] = X_scaler
    pickle.dump(dist_pickle, open("./svc_pickle.p", "wb"))




cars, notcars = load_images()
train_svc(cars, notcars)
