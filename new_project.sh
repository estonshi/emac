# !/bin/bash

set -e

if [ -z $1 ];then
	echo "This script recieves a folder path and create emac project inside."
	echo "Usage : ./emac.new_project <project_folder_path>"
	exit 0
fi

proj_path=$1
mkdir $proj_path/input $proj_path/output
cp ./config.ini $proj_path/config.ini

echo "Done."
