import os
import sys
import shutil
import tarfile
import requests
import json
import random
import struct
import imghdr
import xml.etree.ElementTree as ET

# Wordnet ID for bboxes to extract
wnid_list = [
	'n03642806', 'n03001627', 'n03179701', 'n04379964', 'n03842156',
	'n04190052', 'n03085013', 'n03793489', 'n03782006', 'n03995265', 
	'n03180011', 'n04004767', 'n03222318', 'n04589593']
	
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

# Helper method to get image dims
def get_image_size(fname):
    #Determine the image type of fhandle and return its size.
    #from draco
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height	

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
			response = requests.get(map_url, timeout=30.)
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
				retries = 5
	print("URL map downloaded")

	# Parse file mapping url and download images, throw away files if they
	# do not have corresponding bounding box files or if their URLs have deprecated
	print("Attempting to download {} images for synset: {}...".format(len(image_map), wnid))
	for ix, line in enumerate(image_map):
		prog = int(20. * (ix + 1) / len(image_map))
		sys.stdout.write("\r{}/{} - [{}>{}]".format(ix + 1, len(image_map), "=" * prog, "." * (20 - prog)))
		sys.stdout.flush()
		if line == '':
			continue
		
		line = line.split(' ')
		if "Annotation/{}/{}.xml".format(wnid, line[0]) in files:
			# Try to download image
			img_url = line[1]
			try:
				response = requests.get(img_url, timeout=1., stream=True)
				if response.history:
					# URL was redirected, skip
					continue
				if not 'image' in response.headers['content-type']: 
					# URL is not an image, skip
					continue
				# URL was valid, download image and add ID to valid IDs
				img_file_path = os.path.join(out_dir, "{}.jpg".format(line[0]))
				with open(img_file_path, 'wb') as out_file:
					shutil.copyfileobj(response.raw, out_file)
					valid_image_ids.append(line[0])
				del response
			except:
				pass
				# Bad URL, skip and continue
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
for ix, id in enumerate(valid_image_ids):

	prog = int(20. * (ix + 1) / len(valid_image_ids))
	sys.stdout.write("\r{}/{} - [{}>{}]".format(ix + 1, len(image_map), "=" * prog, "." * (20 - prog)))
	sys.stdout.flush()
	
	src_image_path = os.path.join(out_dir, "{}.jpg".format(id))
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
	
	# Verify bounding box data, throw away image if bbox exceeds image dims
	img_w, img_h = get_image_size(src_image_path)
	bbox_x = int(bbox[0].text)
	bbox_y = int(bbox[1].text)
	bbox_w = int(bbox[2].text) - int(bbox[0].text)
	bbox_h = int(bbox[3].text) - int(bbox[1].text)
	if (bbox_x + bbox_w >= img_w) or (bbox_y + bbox_h >= img_h):
		if id in test_ids:
			test_ids.remove(id)
		else:
			train_ids.remove(id)
		os.remove(src_image_path)
	else:
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
		shape_att['x'] = bbox_x
		shape_att['y'] = bbox_y
		shape_att['width'] = bbox_w
		shape_att['height'] = bbox_h
		region_att = {}
		region_att['object_name'] = id_spl[0]
		region['shape_attributes'] = shape_att
		region['region_attributes'] = region_att
		regions['a'] = region
		img['regions'] = regions
		
		# Write image info to the corresponding dict and move image to the proper folder

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
print("Execution finished, final test/train split:\ntrain[{}]\ttest[{}]".format(len(train_ids), len(test_ids)))
