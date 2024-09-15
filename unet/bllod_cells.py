
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_augmented_images(image_dir, mask_dir, image_size=(512, 512)):
    images = []
    masks = []

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        # Load image
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = load_img(img_path, target_size=image_size, color_mode='grayscale')
        image = img_to_array(image) / 255.0  # Normalize to [0, 1]
        images.append(image)

        # Load mask
        mask = load_img(mask_path, target_size=image_size, color_mode='grayscale')
        mask = img_to_array(mask) / 255.0  # Normalize to [0, 1]
        masks.append(mask)

    return np.array(images), np.array(masks)


from keras import backend as K
import tensorflow as tf

def dice_coef(a, b):
    """Compute the Dice coefficient between two tensors."""
    hi = K.flatten(a)
    pred = K.flatten(b)
    upper_part = 2 * K.sum(hi * pred)
    lower_part = K.sum(hi) + K.sum(pred)
    dice = upper_part / (lower_part + K.epsilon())  # Adding K.epsilon() to avoid division by zero
    return dice

def dice_coef_loss(a, b):
    """Compute the Dice coefficient loss."""
    return 1 - dice_coef(a, b)

def jaccard_index(a, b):
    """Compute the Jaccard index (Intersection over Union)."""
    hi = K.flatten(a)
    pred = K.flatten(b)
    numerator = K.sum(hi * pred)
    denominator = K.sum((hi + pred) - (hi * pred))
    iou = numerator / (denominator + K.epsilon())  # Adding K.epsilon() to avoid division by zero
    return iou


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model

def unet_model(input_size=(512, 512, 1)):
    inputs = Input(input_size)

    # Encoder (Downsampling path)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder (Upsampling path)
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Instantiate the U-Net model
model = unet_model()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.5,
    staircase=True)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

#model.compile(optimizer='adam', loss='binary_crossentropy')
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[dice_coef, jaccard_index])
# Print the model summary
model.summary()


# Load augmented images and masks
image_dir = '../augmented_image1'
mask_dir = '../augmented_mask1'

images, masks = load_augmented_images(image_dir, mask_dir)

# Reshape the masks to fit the model's expected output shape
masks = masks.reshape(masks.shape[0], masks.shape[1], masks.shape[2], 1)  # Add channel dimension if missing





import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Assuming your model (U-Net) is defined as 'model'
# Ensure the augmented_images and augmented_masks are loaded correctly

# Path to save the best model
filepath = "model.keras"

# Define EarlyStopping to stop training if validation loss doesn't improve
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Define ModelCheckpoint to save the best model (based on validation loss)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# List of callbacks
callbacks_list = [earlystopper, checkpoint]

# Train the model
history = model.fit(images,masks,
                    validation_split=0.2,  # 20% of the data will be used for validation
                    batch_size=1,          # You may want to increase this depending on your GPU
                    epochs=20,             # Number of epochs
                    callbacks=callbacks_list)
