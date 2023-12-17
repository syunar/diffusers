accelerate launch --mixed_precision="fp16" --multi_gpu examples/controlnet/train_controlnet.py \
    --pretrained_model_name_or_path="Linaqruf/anything-v3.0" \
    --output_dir="model_out_v3" \
    --dataset_name="/home/users/fronk/projects/deepcolor/datasets/greyscale_manga_fullsize_with_caption" \
    --image_column=image \
    --conditioning_image_column=image \
    --caption_column=text \
    --resolution=768 \
    --learning_rate=1e-5 \
    --max_train_steps=90000 \
    --validation_image "/home/users/fronk/dev/diffusers/018.jpg" "/home/users/fronk/dev/diffusers/020.jpg" \
    --validation_prompt "hetero, sex, comic, doggystyle, skirt, 1boy, cum, sex_from_behind, english_text, penis, 1girl, breasts, pleated_skirt, nipples, vaginal, takarada_rikka, school_uniform, long_hair, bar_censor, multiple_boys" "bar_censor, censored, oral, penis, fellatio, letterboxed, pointless_censoring, licking_penis, hetero, 1boy, school_uniform, identity_censor, blush, nose_blush, multiple_girls, comic, 2girls, short_hair, squatting, bow, partially_colored" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=26 \
    --mixed_precision="fp16" \
    --tracker_project_name="controlnet-greyscale_v3" \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=5000 \
    --validation_steps=5 \
    --proportion_empty_prompts=0.5 \
    --report_to wandb
    # --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    # --num_train_epochs=20 \
    # --dataset_name="/home/users/fronk/dev/diffusers/greyscale_manga250k" \
    # --controlnet_model_name_or_path="lllyasviel/control_v11p_sd15_lineart" \
    # --controlnet_model_name_or_path="ioclab/control_v1p_sd15_brightness" \
