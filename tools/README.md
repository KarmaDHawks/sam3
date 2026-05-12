Readme per spiegare come usare SAM3 per fare inferenza

# ATTIVAZIONE VIRTUAL ENVIRORMENT 

Attivare il venv con il seguente comando:
*conda activate /media/TBData/marco/my_venv/sam3*

# INFERENZA CON SAM3

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