# NOTE: The current version of this script requires you to download a tar.gz
# of ALL bounding box annotations from http://image-net.org/Annotation/Annotation.tar.gz
# Extract it into an Annotations/wnid_uid.tar/wnid_uid.tar structure

# TODO:
# 1. Replace hard coded variables with command line arguments
# 2. Add support for downloading multiple wnid's at once
# 
# Other:
# 1. Allow code to pull files from original tar.gz without unpacking the whole
#    thing. (Unpacked version is 1.22 gb, packed version is 42 mb)

import os
import shutil
import tarfile
import requests
import json
import xml.etree.ElementTree as ET

# Wordnet ID for bboxes to extract
wnid = 'n00007846'

# Define URL paths
map_url = "http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid={}".format(wnid)

# Define direcotry paths
root_dir = os.path.abspath("")
temp_dir = os.path.join(root_dir, ".temp")
cmp_file_path = os.path.join(root_dir, "annotations/{}.tar.gz".format(wnid))
ext_dir = os.path.join(temp_dir, "Annotation/{}".format(wnid))
out_dir = os.path.join(root_dir, "out")
if not os.path.exists(out_dir):
	os.mkdir(out_dir)

# Extract compressed bboxes to temporary dir
annos = tarfile.open(cmp_file_path, 'r:gz')
annos.extractall(temp_dir)

# Generate list of file paths
files = os.listdir(ext_dir)

# Open URL mapping annotation file names to images
response = requests.get(map_url)
image_map = response.text.split("\n")
del response

# Parse file mapping url and download images, throw away files if they
# do not have corresponding bounding box files or if their URLs have deprecated
valid_image_ids = []
for line in image_map:
	if line == '':
		continue
	else:
		line = line.split(' ')
		if "{}.xml".format(line[0]) in files:
			# Try to download image
			img_url = line[1]
			response = requests.get(img_url, stream=True)
			if response.history:
				# URL was redirected, original URL no longer valid
				print("Tried to download {} but the URL is no longer valid...".format(line[0]))
			else:
				# URL was valid, download image and add ID to valid IDs
				img_file_path = os.path.join(out_dir, "{}.jpg".format(line[0]))
				with open(img_file_path, 'wb') as out_file:
					shutil.copyfileobj(response.raw, out_file)
					valid_image_ids.append(line[0])
			del response

# Go through valid ID's annotations to build a JSON file for the dataset
json_file_path = os.path.join(out_dir, "via_region_data.json")
with open(json_file_path, 'w') as json_file:
	images_dict = {}
	for id in valid_image_ids:
		
		# Build JSON
		json_path = os.path.join(out_dir, "via_region_data.json")
		xml_path = os.path.join(ext_dir, "{}.xml".format(id))
		
		tree = ET.parse(xml_path)
		root = tree.getroot()
		bbox = root[5][4]
		print("({}, {}) - ({}, {})".format(bbox[0].text, bbox[1].text, bbox[2].text, bbox[3].text))
			
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
		region_att['object_name'] = wnid
		region['shape_attributes'] = shape_att
		region['region_attributes'] = region_att
		regions['a'] = region
		img['regions'] = regions
		images_dict[id] = img
	
	print(json.dumps(images_dict, indent=2))
	json_file.write(json.dumps(images_dict))

# Remove temp directory
shutil.rmtree(temp_dir)