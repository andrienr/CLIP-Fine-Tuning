#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

The script can be used to train CLIP like models for languages other than English by using
a text encoder pre-trained in the desired language. Currently this script supports the following vision
and text models:
Vision models: ViT(https://huggingface.co/models?filter=vit), CLIP (https://huggingface.co/models?filter=clip)
Text models: BERT, ROBERTa (https://huggingface.co/models?filter=fill-mask)
"""

import logging
import os
import json
import datetime
import shutil
from dataclasses import dataclass, field
from typing import Optional
import tensorflow as tf
from pathlib import Path
from transformers import (
    AutoImageProcessor,
    HfArgumentParser,
    TFBertTokenizer,
    TFTrainingArguments,
    TFVisionTextDualEncoderModel,
    create_optimizer,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import tensorflow_addons as tfa


# Will error if the minimal version of Transformers is not installed. Remove at your own risks
check_min_version("4.30.0.dev0")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    vision_model_name_or_path: str = field(
        default='google/vit-base-patch16-224',
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    text_model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={
            "help": "Path to pretrained image model or model identifier from huggingface.co/models"},
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    images_dir: str = field(
        metadata={"help": "image path"}
    )
    image_path_to_caption_file: str = field(
        metadata={"help": "The input data file (a jsonlines file)."}
    )
    cosine_decay: bool = field(
        metadata={"help": "cosine decay lr"}
    )
    target_shape: Optional[str] = field(
        default=(200*200),
        metadata={"help": "target shape"},
    )
    train_split: Optional[str] = field(
        default=.8,
        metadata={"help": "train split"},
    )


def load_img_text_pairs(image_path_to_caption_file, images_dir):
    with open(image_path_to_caption_file, 'r') as f:
        image_path_to_caption_file = json.load(f)
    captions = []
    images = []
    for image_path in image_path_to_caption_file.keys():
        caption_list = image_path_to_caption_file[image_path]
        captions.extend(caption_list)
        images.extend([os.path.join(images_dir, image_path)]
                      * len(caption_list))
    return (images, captions)


def standardize_text(inputs):
    return tf.strings.regex_replace(inputs,
                                    r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")


def crop_to_square(image):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    if height > width:
        image = tf.image.crop_to_bounding_box(
            image, (height - width) // 2, 0, width, width)
    elif width > height:
        image = tf.image.crop_to_bounding_box(
            image, 0, (width - height) // 2, height, height)
    return image


def create_tf_datasets(image_path_to_caption_file, images_dir, image_processor, caption_tokenizer, train_batch_size=64, test_batch_size=32, train_split=.8):
    images, captions = load_img_text_pairs(
        image_path_to_caption_file, images_dir)
    ds = standardize_text(captions)
    ds = caption_tokenizer(captions)
    del ds['token_type_ids']
    ds['image_path'] = images
    tf_dataset = tf.data.Dataset.from_tensor_slices(ds)

    def load_image(sample):
        image_path = sample['image_path']
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(
            image, channels=3, expand_animations=False)
        image = crop_to_square(image)
        image = tf.image.resize(
            image, list(image_processor.size.values()), method="bicubic", antialias=True)
        image = image / 255.0
        image = (image - image_processor.image_mean) / \
            image_processor.image_std
        # Convert to channels-first
        image = tf.transpose(image, perm=[2, 0, 1])
        sample["pixel_values"] = image
        del sample['image_path']
        return sample

    def transform(sample, seed):
        image = load_image(sample)['pixel_values']
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0]
        # Random brightness.
        image = tf.image.stateless_random_brightness(
            image, max_delta=0.5, seed=new_seed)
        # Random contrast.
        image = tf.image.stateless_random_contrast(
            image, lower=0.1, upper=0.9, seed=new_seed)
        image = tf.image.stateless_random_crop(
            image, size=[3, 224, 224], seed=seed)
        sample["pixel_values"] = image
        return sample

    # Create a generator.
    rng = tf.random.Generator.from_seed(123, alg='philox')

    # Create a wrapper function for updating seeds.
    def update_seed(x):
        seed = rng.make_seeds(2)[0]
        image = transform(x, seed)
        return image

    train_len = round(len(tf_dataset) * float(train_split))
    shuffled = tf_dataset.shuffle(
        len(tf_dataset), seed=42, reshuffle_each_iteration=False)
    train = shuffled\
        .take(train_len)\
        .map(update_seed, num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(train_batch_size)\
        .prefetch(tf.data.AUTOTUNE)
    test = shuffled\
        .skip(train_len)\
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(test_batch_size)\
        .prefetch(tf.data.AUTOTUNE)

    return train, test


def main():

    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        # or by passing the --help flag to this script.
        (ModelArguments, DataTrainingArguments, TFTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Load  processor

    processor = AutoImageProcessor.from_pretrained(
        model_args.vision_model_name_or_path)
    tokenizer = TFBertTokenizer.from_pretrained(
        model_args.text_model_name_or_path)

    # 3. Create datasets

    train_dataset, val_dataset = create_tf_datasets(data_args.image_path_to_caption_file,
                                                    data_args.images_dir,
                                                    processor,
                                                    tokenizer,
                                                    training_args.per_device_train_batch_size,
                                                    training_args.per_device_eval_batch_size,
                                                    data_args.train_split
                                                    )

    # 4. Initialize model

    model = TFVisionTextDualEncoderModel.from_vision_text_pretrained(
        text_model_name_or_path=model_args.text_model_name_or_path,
        vision_model_name_or_path=model_args.vision_model_name_or_path
    )

    if model_args.freeze_vision_model:
        model.vision_model.trainable = False

    if model_args.freeze_text_model:
        model.text_model.trainable = False

    # 5. Training

    if training_args.do_train:
        if data_args.cosine_decay:
            num_train_steps = int(len(train_dataset) *
                                  int(training_args.num_train_epochs))
            initial_learning_rate = 0.0
            decay_steps = int(len(train_dataset) * 90)
            alpha = 0
            warmup_steps = int(len(train_dataset) * 10)
            target_learning_rate = 5e-4
            lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps,
                alpha=alpha,
                name='cosine_decay_no_restart',
                warmup_steps=warmup_steps,
                warmup_target=target_learning_rate,
            )
            optimizer = tfa.optimizers.AdaBelief(
                learning_rate=lr_warmup_decayed_fn)
        else:
            num_train_steps = int(len(train_dataset) *
                                  int(training_args.num_train_epochs))
            if training_args.warmup_steps > 0:
                num_warmup_steps = training_args.warmup_steps
            elif training_args.warmup_ratio > 0:
                num_warmup_steps = int(
                    num_train_steps * training_args.warmup_ratio)
            else:
                num_warmup_steps = 0
            optimizer, lr_schedule = create_optimizer(
                init_lr=training_args.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                adam_beta1=training_args.adam_beta1,
                adam_beta2=training_args.adam_beta2,
                adam_epsilon=training_args.adam_epsilon,
                weight_decay_rate=training_args.weight_decay,
                adam_global_clipnorm=training_args.max_grad_norm,
            )

        model.compile(optimizer=optimizer, jit_compile=training_args.xla)

        # Configure checkpoints

        checkpoint_dir = Path(training_args.output_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = Path(training_args.output_dir)/'cp-{epoch:03d}.ckpt'
        ckpt = tf.train.Checkpoint()
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_dir, max_to_keep=1)
        start_epoch = 0
        if len(os.listdir(checkpoint_dir)) > 0:
            start_epoch = int(
                ckpt_manager.latest_checkpoint.split('-')[-1].split('.')[-2])
            model.load_weights(ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            verbose=1
        )

        # Configure training logs

        LOG_DIR = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR, histogram_freq=1)

        if not training_args.do_eval:
            val_dataset = None

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            initial_epoch=start_epoch,
            epochs=int(training_args.num_train_epochs),
            callbacks=[model_checkpoint_callback, tensorboard_callback]
        )

        # Save text model
        SAVED_MODELS_DIR = Path('saved_models')
        SAVED_MODELS_DIR.mkdir(exist_ok=True)
        if len(os.listdir(SAVED_MODELS_DIR)) > 0:
            version = os.listdir(SAVED_MODELS_DIR)
            version.sort()
            version = int(version[-1])+1
        else:
            version = 1
        export_path = os.path.join(SAVED_MODELS_DIR, str(version))
        tf.keras.models.save_model(
            model,
            export_path,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )
        print('\nModel saved')

    # # 10. Evaluation

    if training_args.do_eval and not training_args.do_train:
        model.evaluate(val_dataset)


if __name__ == "__main__":
    main()
