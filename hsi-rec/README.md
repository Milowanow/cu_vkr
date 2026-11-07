```bash
 uv run python utils/create_rgb.py --source-mat ../Dataset/NTIRE/hsi/ARAD_1K_0001.mat --save-path ../Dataset/NTIRE/jpg/  --camspec-path ../Dataset/camspec_database.txt  

 uv run python utils/hsi_to_rgb_pipeline.py --list-file ./dataset_files_lists/NTIRE_files.txt --download-dir ../Dataset/NTIRE/hsi --rgb-dir ../Dataset/NTIRE/jpg   --start 0
 ```