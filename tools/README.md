Readme per spiegare come usare SAM3 per fare inferenza

# ATTIVAZIONE VIRTUAL ENVIRORMENT 

Attivare il venv con il seguente comando:
*conda activate /media/TBData/marco/my_venv/sam3*

# INFERENZA CON SAM3 su dataset VOS like

python tools/vos_inference.py \
--base_video_dir /path-alla-cartella-dei-frame/ \
--input_mask_dir /path-alla-cartella-delle-maschere \
--video_list_file /path-alla-lista-dei-video/val.txt \
--output_mask_dir /path-di-dove-salvare-output/

# PARAMETRI OPZIONALI
--offload_video_to_cpu
    Per offloadare i video alla CPU e risparmiare memoria GPU (nel caso di video lunghi)


Cartella outputs contiene i risultati delle inferenze sui vari benchmark VOS
*/home/marco/Desktop/SAM3-exp/sam3/outputs*

CUDA_VISIBLE_DEVICES=4 python ./tools/vos_inference.py \
--base_video_dir /media/TBData2/data/VOST/SingleObject/JPEGImages \
--input_mask_dir /media/TBData2/data/VOST/SingleObject/Annotations \
--video_list_file /media/TBData2/data/VOST/SingleObject/ImageSets/val.txt \
--output_mask_dir /home/marco/Desktop/SAM3-exp/sam3/outputs/test/VOST_val_clean \
--offload_video_to_cpu

# Inferenza SAM3 per image segmentation con prompt testuali

# Esempio di utilizzo completo (salva sia PNG che JSON)
python image_text_prompt_inference.py \
  --dataset_dir /path/to/dataset \
  --prompt "cup" \
  --output_mask_dir ./results/masks \
  --output_json_dir ./results/coordinates

# Solo maschere PNG
python image_text_prompt_inference.py \
  --dataset_dir /path/to/dataset \
  --prompt "person" \
  --output_mask_dir ./results/masks \
  --no_json

# Solo coordinate JSON
python image_text_prompt_inference.py \
  --dataset_dir /path/to/dataset \
  --prompt "shoe" \
  --output_json_dir ./results/coordinates \
  --no_masks