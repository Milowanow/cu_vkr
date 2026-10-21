```bash

uv run python get_data.py --remote-path ICVL_our/ulm_0328-1118.mat --save-path ./downloaded
uv run python get_data.py --remote-path NTIRE/ARAD_1K_0945.mat --save-path ./downloaded/NTIRE/


uv run python create_rgb.py --source-mat ../downloaded/ICVL_our/4cam_0411-1640-1.mat --save-path ../raw/ICVL_our/
uv run python hsi_to_rgb_pipeline.py --list-file ./ICVL_our_files.txt --download-dir ../downloaded/ICVL_our/ --rgb-dir ../rgb   --start 0
```



#Установка uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env


#отправить файл скредами
scp ./cred.json root@185.182.108.29:/cred.json

uv sync