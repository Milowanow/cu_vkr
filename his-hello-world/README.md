```bash

uv run python get_data.py --remote-path ICVL_our/ulm_0328-1118.mat --save_path ./downloaded
uv run python get_data.py --remote-path NTIRE/ARAD_1K_0945.mat --save-path ./downloaded/NTIRE/


uv run python process_data.py --source-mat downloaded/ulm_0328-1118.mat --save-path ./raw
```