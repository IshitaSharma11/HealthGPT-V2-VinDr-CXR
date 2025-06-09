!git clone https://github.com/DCDmllm/HealthGPT.git
%cd HealthGPT


##  !pip install -r requirements.txt


!mkdir -p llava/clip_model
!mkdir -p llava/hlora_weights
!mkdir -p llava/fusion_layer
!mkdir -p taming_transformers/ckpt


!wget https://huggingface.co/lintw/HealthGPT-M3/resolve/main/com_hlora_weights.bin -P llava/hlora_weights/
!wget https://huggingface.co/lintw/HealthGPT-M3/resolve/main/fusion_layer_weights.bin -P llava/fusion_layer/


!wget -O taming_transformers/ckpt/last.ckpt "https://heibox.uni-heidelberg.de/f/867b05fc8c481768640/?dl=1"


!wget -O taming_transformers/ckpt/model111.yaml "https://heibox.uni-heidelberg.de/f/274fb24ed3831bfa753/?dl=1"


!wget -O llava/clip_model/pytorch_model.bin "https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/pytorch_model.bin"


!wget -O llava/clip_model/config.json "https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/config.json"




!wget -O llava/clip_model/preprocessor_config.json "https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/preprocessor_config.json"


from google.colab import files
uploaded = files.upload()



!python llava/demo/com_infer.py \
  --model_name_or_path "microsoft/Phi-3-mini-4k-instruct" \
  --dtype "FP16" \
  --hlora_r 64 \
  --hlora_alpha 128 \
  --hlora_nums 4 \
  --vq_idx_nums 8192 \
  --instruct_template "phi3_instruct" \
  --vit_path "llava/clip_model" \
  --hlora_path "llava/hlora_weights/com_hlora_weights.bin" \
  --fusion_layer_path "llava/fusion_layer/fusion_layer_weights.bin" \
  --question "What abnormalities are visible in this chest X-ray?" \
  --img_path "011111.jpg"
