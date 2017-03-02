#single image tests
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
from window_utils import search_windows, slide_window, draw_boxes, find_cars, draw_labeled_bboxes, apply_threshold
from feature_utils import single_img_features
from train_svc import load_images
from utils import visualize
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from heatmap_history import heatmap_history
from labels_history import labels_history

def single_image_test(cars, notcars, y_start_stop, color_space, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat):
    #choose random car / not car indices
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    #Read in car / not car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    car_features, car_hog_image = single_img_features(car_image, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat, vis=True)
    notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

    images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
    titles = ['car image', 'car HOG image', 'notcar image', 'notcar HOG image']
    fig = plt.figure(figsize=(12,3))
    visualize(fig, 1, 4, images, titles)


def video_images_test(y_start_stop, overlap, scale, color_space, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat):
    searchpath = 'test_images/*'
    example_images = glob.glob(searchpath)
    images = []
    titles = []

    for img_src in example_images:
        t1 = time.time()
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        print(np.min(img), np.max(img))

        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(128,128), xy_overlap=(overlap, overlap))
        hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

        window_img = draw_boxes(draw_img, hot_windows, color=(0,0,255), thick=6)
        images.append(window_img)
        titles.append('')
        print(time.time()-t1, 'seconds to process one image searching', len(windows), 'windows')
    fig = plt.figure(figsize=(12,2))
    visualize(fig, 1, 6, images, titles)

def video_images_test_single_hog(heatmap_history, start, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    searchpath = 'test_images/*'
    example_images = glob.glob(searchpath)

    out_images = []
    out_maps = []
    out_titles = []
    out_boxes = []

    for img_src in example_images:
        img = mpimg.imread(img_src)
        heatmap = np.zeros_like(img[:,:,0])

        scale = 1
        y_start_stop = [400, 528]
        heatmap = find_cars(img, heatmap, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        scale = 1.5
        y_start_stop = [400, 656]
        heatmap = find_cars(img, heatmap, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        scale = 2
        y_start_stop = [400, 656]
        heatmap = find_cars(img, heatmap, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        #avergaed_heatmap = heatmap_history.averaged_heatmap(heatmap)

        avergaed_heatmap = apply_threshold(heatmap, 2)

        labels = label(avergaed_heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        out_images.append(draw_img)
        out_images.append(labels[0])
        #fig = plt.figure(figsize=(24,2))
        #visualize(fig, 2, 6, out_images, out_titles)

    fig = plt.figure(figsize=(24,2))
    visualize(fig, 2, 6, out_images, out_titles)

def process_image(img, heatmap_history):
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
    dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["X_scaler"]
    heatmap = np.zeros_like(img[:,:,0])

    scale = 1
    y_start_stop = [400, 528]
    heatmap = find_cars(img, heatmap, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    scale = 1.5
    y_start_stop = [400, 656]
    heatmap = find_cars(img, heatmap, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    scale = 2
    y_start_stop = [400, 656]
    heatmap = find_cars(img, heatmap, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    #combine with heatmaps from previous frames
    averaged_heatmap = heatmap_history.averaged_heatmap(heatmap)
    averaged_heatmap = apply_threshold(averaged_heatmap, 6)
    labels = label(averaged_heatmap)


    #heatmap_thresholded = apply_threshold(heatmap, 2)
    #labels = label(heatmap)
    #label_heatmap = labels_history.averaged_labels(labels[0])
    #threshold_label_heatmap = apply_threshold(label_heatmap, 3)
    #labels = label(label_heatmap)


    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

def process(img):
    processed = process_image(img, heatmap_history)
    return processed

def process_video():
    #test_output = 'test_ouput.mp4'
    test_output = 'project_ouput2.mp4'
#    clip = VideoFileClip('test_video.mp4')
    clip = VideoFileClip('project_video.mp4')
    test_clip = clip.fl_image(process)
    test_clip.write_videofile(test_output, audio=False)




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
dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
y_start_stop = [400, 656]
scale = 1.5
overlap = .5
heatmap_history = heatmap_history()
labels_history = labels_history()
#cars, notcars = train_svc()
#single_image_test(cars, notcars, y_start_stop, color_space, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat)
#video_images_test(y_start_stop, overlap, scale, color_space, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat)
#video_images_test_single_hog(heatmap_history, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

process_video()
