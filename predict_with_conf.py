# similar to predict.py but generates prediction for the face with high confidence score only visible in an image
# predict.py (Modified)
from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import dlib
import os
import argparse

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def detect_face(image_paths, SAVE_DETECTED_AT, default_max_size=800, size=300, padding=0.25):
	cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
	sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
	
	for index, image_path in enumerate(image_paths):
		if index % 1000 == 0:
			print('--- Detecting faces: %d/%d ---' % (index, len(image_paths)))
		
		img = dlib.load_rgb_image(image_path)
		old_height, old_width, _ = img.shape
		
		if old_width > old_height:
			new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
		else:
			new_width, new_height = int(default_max_size * old_width / old_height), default_max_size
		img = dlib.resize_image(img, rows=new_height, cols=new_width)
		
		dets = cnn_face_detector(img, 1)
		
		if len(dets) == 0:
			print("Sorry, there were no faces found in '{}'".format(image_path))
			continue
		
		# --- MODIFICATION START ---
		# Find the detection with the highest confidence score.
		# This ensures we only process the most prominent face in the image.
		best_detection = max(dets, key=lambda det: det.confidence)
		# --- MODIFICATION END ---
		
		faces = dlib.full_object_detections()
		faces.append(sp(img, best_detection.rect))
		
		# --- MODIFICATION START ---
		# Extract only the single best face
		image = dlib.get_face_chips(img, faces, size=size, padding=padding)[0]
		
		# Create a clean filename that can be traced back to the original
		original_img_name = os.path.basename(image_path)
		# Save the single cropped face
		face_name = os.path.join(SAVE_DETECTED_AT, original_img_name)
		dlib.save_image(image, face_name)
		# --- MODIFICATION END ---

def predidct_age_gender_race(save_prediction_at, imgs_path='cropped_faces/'):
    # ... (This function remains mostly the same, but we will modify the output part)
	img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model_fair_7 = torchvision.models.resnet34(pretrained=True)
	model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
	model_fair_7.load_state_dict(torch.load('dlib_models/res34_fair_align_multi_7_20190809.pt'))
	model_fair_7 = model_fair_7.to(device)
	model_fair_7.eval()

	model_fair_4 = torchvision.models.resnet34(pretrained=True)
	model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
	model_fair_4.load_state_dict(torch.load('dlib_models/res34_fair_align_multi_4_20190809.pt'))
	model_fair_4 = model_fair_4.to(device)
	model_fair_4.eval()

	trans = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	
	# ... (prediction loop is identical) ...
	original_image_paths = []
	race_scores_fair, gender_scores_fair, age_scores_fair = [], [], []
	race_preds_fair, gender_preds_fair, age_preds_fair = [], [], []
	race_scores_fair_4, race_preds_fair_4 = [], []

	for index, img_name in enumerate(img_names):
		if index % 1000 == 0:
			print("Predicting... {}/{}".format(index, len(img_names)))
		
		# --- MODIFICATION START ---
		# The filename of the cropped face IS the original filename now
		original_image_paths.append(img_name)
		# --- MODIFICATION END ---
		
		image = dlib.load_rgb_image(img_name)
		image = trans(image)
		image = image.view(1, 3, 224, 224)
		image = image.to(device)

		# Fair 7-class
		outputs = model_fair_7(image)
		outputs = outputs.cpu().detach().numpy().squeeze()
		race_outputs, gender_outputs, age_outputs = outputs[:7], outputs[7:9], outputs[9:18]
		race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
		gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
		age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))
		race_preds_fair.append(np.argmax(race_score))
		gender_preds_fair.append(np.argmax(gender_score))
		age_preds_fair.append(np.argmax(age_score))
		race_scores_fair.append(race_score)
		gender_scores_fair.append(gender_score)
		age_scores_fair.append(age_score)

		# Fair 4-class
		outputs = model_fair_4(image)
		outputs = outputs.cpu().detach().numpy().squeeze()
		race_outputs = outputs[:4]
		race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
		race_preds_fair_4.append(np.argmax(race_score))
		race_scores_fair_4.append(race_score)
	
	# --- MODIFICATION START ---
	# Add the original image name to the dataframe
	result = pd.DataFrame([original_image_paths, race_preds_fair, race_preds_fair_4, gender_preds_fair, age_preds_fair]).T
	result.columns = ['original_image', 'race_preds_fair', 'race_preds_fair_4', 'gender_preds_fair', 'age_preds_fair']
	# --- MODIFICATION END ---
	
	# ... (the rest of the DataFrame processing for labels is the same) ...
	result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
	result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
	result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
	result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
	result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
	result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
	result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'
	result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
	result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'
	result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
	result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
	result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
	result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
	result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
	result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
	result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
	result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
	result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

	# Save final results
	result[['original_image', 'race', 'gender', 'age']].to_csv(save_prediction_at, index=False)
	print("saved results at ", save_prediction_at)

# ... (main execution block is the same) ...
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', dest='input_csv', action='store', help='csv file of image path where col name for image path is "img_path')
    dlib.DLIB_USE_CUDA = True
    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
    args = parser.parse_args()
    SAVE_DETECTED_AT = "detected_faces"
    ensure_dir(SAVE_DETECTED_AT)
    imgs = pd.read_csv(args.input_csv)['img_path']
    detect_face(imgs, SAVE_DETECTED_AT)
    print("detected faces are saved at ", SAVE_DETECTED_AT)
    predidct_age_gender_race("test_outputs.csv", SAVE_DETECTED_AT)
