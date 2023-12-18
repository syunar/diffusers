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
    --validation_image "validation_images/228626/019.jpg" "validation_images/228626/013.jpg" "validation_images/228626/026.jpg" "validation_images/228626/014.jpg" "validation_images/228626/017.jpg" \
    --validation_prompt "1boy, 1girl, anus, ass, bar_censor, blush, breasts, censored, clothed_sex, comic, deep_penetration, female_pubic_hair, hetero, male_pubic_hair, mating_press, penis, plaid, plaid_skirt, pointless_censoring, pubic_hair, pussy, sex, skirt, testicles, vaginal" "1boy, 1girl, blush, bra, bra_lift, breast_grab, breasts, closed_eyes, clothes_lift, comic, grabbing, hetero, large_breasts, nipple_tweak, nipples, open_mouth, paizuri, shirt_lift, short_hair, smile, speech_bubble, underwear" "1boy, 1girl, after_sex, bar_censor, blush, breasts, censored, cervix, comic, cross-section, cum, cum_in_pussy, deep_penetration, ejaculation, hetero, internal_cumshot, medium_breasts, nipples, nude, open_mouth, overflow, penis, pointless_censoring, pussy, sex, short_hair, thighhighs, uterus, vaginal, x-ray" "1boy, 1girl, blush, breast_sucking, breasts, breasts_apart, comic, english_text, hetero, lactation, large_breasts, licking, licking_breast, licking_nipple, nipple_tweak, nipples, open_mouth, pillarboxed, tongue, tongue_out" "1boy, black_hair, breasts, collared_shirt, comic, english_text, head_rest, long_hair, multiple_girls, school_uniform, shirt, sitting, skirt, speech_bubble" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=26 \
    --mixed_precision="fp16" \
    --tracker_project_name="controlnet-greyscale_v3" \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=5000 \
    --validation_steps=1000 \
    --proportion_empty_prompts=0.5 \
    --report_to wandb
    # --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    # --num_train_epochs=20 \
    # --dataset_name="/home/users/fronk/dev/diffusers/greyscale_manga250k" \
    # --controlnet_model_name_or_path="lllyasviel/control_v11p_sd15_lineart" \
    # --controlnet_model_name_or_path="ioclab/control_v1p_sd15_brightness" \
