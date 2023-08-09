#!/bin/sh
rm -r pretrained_models
# download data for fine-tuning
rm sampled_audio4ft_v2.zip
rm -r sampled_audio4ft
# create necessary directories
rm -r video_data
rm -r raw_audio
rm -r denoised_audio
rm -r custom_character_voice
rm -r segmented_character_voice
rm short_character_anno.txt long_character_anno.txt final_annotation_train.txt final_annotation_val.txt sampled_audio4ft.txt
