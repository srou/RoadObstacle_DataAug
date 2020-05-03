# Before running, first run export DIR_OUTPUT=output_dir_path
OUTPUT_DIR="output_dir_cityscapes_rgbd"
rm ./$OUTPUT_DIR/train.txt
rm -v ./$OUTPUT_DIR/annotations/*.xml
rm -v ./$OUTPUT_DIR/annotations_txt/*.txt
rm -v ./$OUTPUT_DIR/annotations_twoclasses_txt/*.txt
rm -v ./$OUTPUT_DIR/images/*.png