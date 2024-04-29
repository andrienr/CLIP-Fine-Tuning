SCRIPT_DIR=.
IMAGES_DIR='../images'
IMAGE_PATH_TO_CAPTION_FILE='image_path_to_caption_train.json'
OUTPUT_DIR='training_output'

python ${SCRIPT_DIR}/task.py \
    --images_dir=${IMAGES_DIR} \
    --image_path_to_caption_file=${IMAGE_PATH_TO_CAPTION_FILE} \
    --output_dir=${OUTPUT_DIR} \
    --freeze_vision_model=True \
    --freeze_text_model=True \
    --do_train=True \
    --do_eval=False \
    --cosine_decay=True \
    --train_split=.001 \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="32" \
    --num_train_epochs=20 \