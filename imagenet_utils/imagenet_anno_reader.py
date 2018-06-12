import os
import sys
import shutil
import tarfile
import requests
import json
import random
import xml.etree.ElementTree as ET

# Wordnet ID for bboxes to extract
wnid_list = ['n00007846', 'n00015388', 'n00017222']

# Train/Test split
train_test_split = 0.15

# Define directory paths
root_dir = os.path.abspath("")
anno_path = os.path.join(root_dir, "Annotation.tar.gz")

out_dir = os.path.join(root_dir, "out")
train_dir = os.path.join(out_dir, "train")
test_dir = os.path.join(out_dir, "val")
if not os.path.exists(out_dir):
	os.mkdir(out_dir)
if not os.path.exists(train_dir):
	os.mkdir(train_dir)
if not os.path.exists(test_dir):
	os.mkdir(test_dir)

# Extract annotation file
annotations = tarfile.open(anno_path)

valid_image_ids = []
last_ix = 0
for wnid in wnid_list:
	
	# Define URL path
	map_url = "http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid={}".format(wnid)

	# Extract relevant annotation file
	target_file = "{}.tar.gz".format(wnid)
	assert target_file in annotations.getnames(), \
		"{} not in annotation file...\nMaybe no annotations exist?".format(wnid)
	subtar = annotations.extractfile(target_file)
	subannos = tarfile.open(fileobj=subtar)
	files = subannos.getnames()

	# Open URL mapping annotation file names to images
	print("Requesting URL map for synset: {}\nURL: {}...".format(wnid, map_url))
	retries = 5
	while retries > 0:
		try:
			response = requests.get(map_url)
			image_map = response.text.split("\n")
			del response
			retries = 0
		except:
			retries -= 1
			print("Connection timeout {}/5 while requesting map...".format(5 - retries))
			if retries == 0:
				user_input = ""
				while user_input.lower() not in ['y', 'n']:
					user_input = input("Try again? (y/n) ")
				if user_input.lower() == 'n':
					raise
	print("URL map downloaded")

	# Parse file mapping url and download images, throw away files if they
	# do not have corresponding bounding box files or if their URLs have deprecated
	print("Attempting to download {} images for synset: {}...".format(len(image_map), wnid))
	for ix, line in enumerate(image_map):
		prog = int(20. * (ix + 1) / len(image_map))
		sys.stdout.write("\r{}/{} - <{}{}>".format(ix + 1, len(image_map), "=" * prog, "." * (20 - prog)))
		sys.stdout.flush()
		if line == '':
			continue
		else:
			line = line.split(' ')
			if "Annotation/{}/{}.xml".format(wnid, line[0]) in files:
				# Try to download image
				img_url = line[1]
				response = requests.get(img_url, stream=True)
				if not response.history:
					# URL was valid, download image and add ID to valid IDs
					img_file_path = os.path.join(out_dir, "{}.jpg".format(line[0]))
					with open(img_file_path, 'wb') as out_file:
						shutil.copyfileobj(response.raw, out_file)
						valid_image_ids.append(line[0])
				del response
	print("\nSuccesfully downloaded {}/{} images...".format(len(valid_image_ids) - last_ix, len(image_map)))
	last_ix = len(valid_image_ids)
print("Succesfully gathered {} images from {} synsets".format(last_ix, len(wnid_list)))

# Shuffle valid_image_ids and generate train and test splits
print("Generating train and test splits...")
random.shuffle(valid_image_ids)
test_ids = random.sample(valid_image_ids, int(len(valid_image_ids) * train_test_split))
train_ids = []
for id in valid_image_ids:
	if id not in test_ids:
		train_ids.append(id)
print("Split sets into:\ntrain[{}]\ttest[{}]".format(len(train_ids), len(test_ids)))
	
# Go through valid ID's annotations to build a JSON file for the dataset
train_json_path = os.path.join(train_dir, "via_region_data.json")
train_dict = {}
test_json_path = os.path.join(test_dir, "via_region_data.json")
test_dict = {}

print("Generating JSON from annotations...")
for id in valid_image_ids:
	id_spl = id.split('_')
	
	target_file = "{}.tar.gz".format(id_spl[0])
	subtar = annotations.extractfile(target_file)
	subannos = tarfile.open(fileobj=subtar)
	
	# Load annotation data
	json_path = os.path.join(out_dir, "via_region_data.json")
	xml_path = "Annotation/{}/{}.xml".format(id_spl[0], id)
	f = subannos.extractfile(xml_path)
	tree = ET.parse(f)
	root = tree.getroot()
	bbox = root[5][4]
	
	# Build dict to convert to JSON
	img = {}
	img['fileref'] = id
	img['size'] = os.path.getsize(os.path.join(out_dir, "{}.jpg".format(id))) 
	img['filename'] = "{}.jpg".format(id)
	img['base64_img_data'] = ""
	img['file_attributes'] = {}
	regions = {}
	region = {}
	shape_att = {}
	shape_att['name'] = "rect"
	shape_att['x'] = int(bbox[0].text)
	shape_att['y'] = int(bbox[1].text)
	shape_att['width'] = int(bbox[2].text) - int(bbox[0].text)
	shape_att['height'] = int(bbox[3].text) - int(bbox[1].text)
	region_att = {}
	region_att['object_name'] = id_spl[0]
	region['shape_attributes'] = shape_att
	region['region_attributes'] = region_att
	regions['a'] = region
	img['regions'] = regions
	
	# Write image info to the corresponding dict and move image to the proper folder
	src_image_path = os.path.join(out_dir, "{}.jpg".format(id))
	if id in test_ids:
		test_dict[id] = img
		dest_image_path = os.path.join(test_dir, "{}.jpg".format(id))
	else:
		train_dict[id] = img
		dest_image_path = os.path.join(train_dir, "{}.jpg".format(id))
	shutil.move(src_image_path, dest_image_path)
print("Finished generating annotations, writing them to the disk...")
	
# Write JSON to their respective files
with open(train_json_path, 'w') as train_json_file:
	with open(test_json_path, 'w') as test_json_file:
		train_json_file.write(json.dumps(train_dict))
		test_json_file.write(json.dumps(test_dict))
