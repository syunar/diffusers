accelerate launch --mixed_precision="fp16" --multi_gpu examples/controlnet/train_controlnet.py \
    --pretrained_model_name_or_path="Linaqruf/anything-v3.0" \
    --output_dir="model_out_v3" \
    --dataset_name="/home/users/fronk/diffusers/data/greyscale_manga_fullsize_with_caption" \
    --image_column=image \
    --conditioning_image_column=image \
    --caption_column=text \
    --max_train_steps=50000 \
    --resolution=768 \
    --learning_rate=1e-5 \
    --train_batch_size=11 \
    --gradient_accumulation_steps=6 \
    --checkpointing_steps=5000 \
    --validation_steps=1000 \
    --proportion_empty_prompts=0.5 \
    --validation_image "validation_images/228626/019.jpg" "validation_images/228626/013.jpg" "validation_images/228626/026.jpg" "validation_images/228626/014.jpg" "validation_images/228626/017.jpg" "validation_images/228626/011.jpg" "validation_images/228626/023.jpg" "validation_images/228626/012.jpg" "validation_images/228626/006.jpg" "validation_images/228626/016.jpg" \
    --validation_prompt "1boy, 1girl, anus, ass, bar_censor, blush, breasts, censored, clothed_sex, comic, deep_penetration, female_pubic_hair, hetero, male_pubic_hair, mating_press, penis, plaid, plaid_skirt, pointless_censoring, pubic_hair, pussy, sex, skirt, testicles, vaginal" "1boy, 1girl, blush, bra, bra_lift, breast_grab, breasts, closed_eyes, clothes_lift, comic, grabbing, hetero, large_breasts, nipple_tweak, nipples, open_mouth, paizuri, shirt_lift, short_hair, smile, speech_bubble, underwear" "1boy, 1girl, after_sex, bar_censor, blush, breasts, censored, cervix, comic, cross-section, cum, cum_in_pussy, deep_penetration, ejaculation, hetero, internal_cumshot, medium_breasts, nipples, nude, open_mouth, overflow, penis, pointless_censoring, pussy, sex, short_hair, thighhighs, uterus, vaginal, x-ray" "1boy, 1girl, blush, breast_sucking, breasts, breasts_apart, comic, english_text, hetero, lactation, large_breasts, licking, licking_breast, licking_nipple, nipple_tweak, nipples, open_mouth, pillarboxed, tongue, tongue_out" "1boy, black_hair, breasts, collared_shirt, comic, english_text, head_rest, long_hair, multiple_girls, school_uniform, shirt, sitting, skirt, speech_bubble" "1boy, 2girls, bar_censor, bikini, black_hair, blush, bottomless, bow, bowtie, breasts, censored, clenched_teeth, comic, cum, hetero, large_breasts, open_mouth, penis, pointless_censoring, ponytail, school_uniform, sex, shirt, short_hair, short_sleeves, skirt, speech_bubble, underwear, vaginal" "1boy, 1girl, assertive_female, bar_censor, blush, breasts, button_gap, censored, cleavage, comic, large_breasts, mole, multiple_girls, nude, open_mouth, paizuri, partially_unbuttoned, penis, pubic_hair, pussy, see-through, sex, shirt, short_hair, sitting, speech_bubble" "1boy, 1girl, bar_censor, blush, bottomless, bow, bowtie, breasts, censored, cervix, closed_eyes, clothed_sex, comic, cross-section, ejaculation, hetero, medium_breasts, open_mouth, penis, pointless_censoring, pussy, school_uniform, sex, shirt, short_hair, speech_bubble, vaginal, x-ray" "1boy, 1girl, black_hair, breasts, bus_stop, collared_shirt, comic, large_breasts, necktie, open_mouth, outdoors, phone_screen, pillarboxed, rain, school_uniform, shirt, short_sleeves, speech_bubble, tree, wet" "2girls, bar_censor, blush, bouncing_breasts, breasts, breasts_apart, censored, cervix, comic, cross-section, cum, cum_in_pussy, cum_on_body, cum_on_breasts, ejaculation, hetero, internal_cumshot, large_breasts, lying, medium_breasts, multiple_girls, navel, nipples, on_back, open_mouth, penis, ponytail, pussy, sex, speech_bubble, spoken_squiggle, vaginal, x-ray" \
    --mixed_precision="fp16" \
    --tracker_project_name="controlnet-greyscale_v3" \
    --enable_xformers_memory_efficient_attention \
    --report_to wandb
    # --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    # --num_train_epochs=20 \
    # --dataset_name="/home/users/fronk/dev/diffusers/greyscale_manga250k" \
    # --controlnet_model_name_or_path="lllyasviel/control_v11p_sd15_lineart" \
    # --controlnet_model_name_or_path="ioclab/control_v1p_sd15_brightness" \
