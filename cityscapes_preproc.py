'''
Moves a selection of Cityscapes leftImg8bit and color label files into
a new directory. 

Output directory structure is designed to be used with our
data augmentation script based on SynDataGeneration.
'''

import sys, os
import shutil
import argparse
import numpy as np
from pathlib import Path, PurePath
from defaults import CITYSCAPES_DIR, BACKGROUND_DIR

class Quiet:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def create_dirs():
	folders = {
		"input_dir_cityscapes_rgbd": ["backgrounds", "distractor_objects_dir", "objects_dir"],
		"objects_dir": ["RGBD_objects"],
	}

	text_files = [
		"input_dir_cityscapes_rgbd/neg_list.txt",
		"input_dir_cityscapes_rgbd/selected.txt",
	]

	for (parent, dirs) in folders.items():
		for directory in dirs:
			(DIR_HERE / parent / directory).mkdir(parents=True, exist_ok=True)

	for file in text_files:
		(DIR_HERE / file).touch()
		if "selected" in file:
			with open(str(DIR_HERE / file), "w") as f:
				f.write("RGBD_objects")

def get_src():
	# Get images from dataset
	src_path = args.src / "leftImg8bit" / args.split
	src_imgls = sorted(src_path.glob("**/*.png"))
	assert len(src_imgls), "No images found"

	# Random permute
	perm_idx = np.random.choice(len(src_imgls), args.num, replace=False)
	perm_imgls = [src_imgls[i] for i in perm_idx]

	return perm_imgls

def get_anno(dsetdir, imgls):
	if not isinstance(dsetdir, PurePath): dsetdir = Path(dsetdir)
	anno_templ = str(dsetdir / "gtFine" / args.split) + "/{parent}/{img_id}_gtFine_color.png"
	annols = []

	for imgpath in imgls:
		annols.append(Path(anno_templ.format(
			parent=imgpath.name.split("_")[0], 
			img_id=imgpath.name[:-16])
		))

	return annols

def movels(subdir, *argv):
	for pathls in argv:
		for ele in pathls:
			split_idx = ele.parts.index(args.split)
			parent = ele.parts[split_idx+1]
			name = ele.parts[-1]
			
			src_str = str(ele)
			if args.keep_subdirs: dst_path = args.dst / parent / name
			else: dst_path = args.dst / name
			if not dst_path.parent.exists(): dst_path.parent.mkdir(parents=True)
			
			dst_str = str(dst_path)
			shutil.copyfile(src_str, dst_str)

def modify_cityscapes(args):
	print("Selectively importing Cityscapes dataset...")
	print("Arguments:")
	print(args)
	args.src = Path(args.src)
	args.dst = Path(args.dst)
	create_dirs()
	imgls = get_src()
	annols = get_anno(args.src, imgls)
	movels("backgrounds", imgls, annols)
	print("Done.")



if __name__ == "__main__":
	DIR_HERE = Path(__file__).resolve().parent
	DIR_BG = Path(DIR_HERE/BACKGROUND_DIR)

	parser = argparse.ArgumentParser(description="Modify Cityscapes for data augmentation")
	parser.add_argument("--num", "-n", type=int, help="Number of images to use", default=300)
	parser.add_argument("--src", "-s", type=str, help="Root directory of Cityscapes dataset", default=str(CITYSCAPES_DIR))
	parser.add_argument("--dst", "-d", type=str, help="Directory in which to save modified dataset", default=str(DIR_BG))
	parser.add_argument("--split", "-sp", type=str, help="Name of string to consider when globbing files", default="train")
	parser.add_argument("--keep_subdirs", "-k", action="store_true", help="Keep original subdirectories", default=False)
	parser.add_argument("--quiet", "-q", action="store_true", help="Suppress print calls", default=False)

	args = parser.parse_args()
	if args.quiet: 
		with Quiet(): 
			modify_cityscapes(args)
	else: modify_cityscapes(args)