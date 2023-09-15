import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from patchify import patchify
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping




#################################### Vision Transformer Architecture - start ###################################################

class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_initialize = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value = w_initialize(shape=(1, 1, input_shape[-1]), dtype=tf.float32), trainable = True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        s = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        s = tf.cast(s, dtype=inputs.dtype)
        return s


def mlp(x, mlp_dim, dropout_rate, hidden_dim):
    x = Dense(mlp_dim, activation="gelu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_dim)(x)
    x = Dropout(dropout_rate)(x)
    return x


def transformer_encoder(x, num_heads, hidden_dim):
    save_x_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)(x, x)
    x = Add()([x, save_x_1])
    save_x_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, mlp_dim, dropout_rate, hidden_dim)
    x = Add()([x, save_x_2])
    return x


def ViT(num_patches, patch_size, num_channels, hidden_dim, num_layers, num_classes):
    input_shape = (num_patches, patch_size*patch_size*num_channels)
    inputs = Input(input_shape)    

    # Patch and position embeddings
    patch_embed = Dense(hidden_dim)(inputs)   
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = Embedding(input_dim=num_patches, output_dim=hidden_dim)(positions) 
    embed = patch_embed + pos_embed         # Add positional embeddings to patches

    # Class token
    token = ClassToken()(embed)
    x = Concatenate(axis=1)([token, embed]) 

    for i in range(num_layers):
        x = transformer_encoder(x, num_heads, hidden_dim)

    # Classification head
    x = LayerNormalization()(x)    
    x = x[: , 0 , :]
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

#################################### Vision Transformer Architecture - end ###################################################





######################################### Training - start ################################################################

image_size     = 224
num_channels   = 3                                                  
patch_size     = image_size // 14                                    # Size of patches extracted from the input image
num_patches    = (image_size // patch_size)**2                       # Number of patches
flat_patches   = (num_patches, patch_size*patch_size*num_channels)

batch_size     = 32
lr             = 1e-4
num_epochs     = 100
num_classes    = 3                                                   # Number of output classes
dropout_rate   = 0.2
class_names    = ["Class A", "Class B", "Class C"]                   # Name of classes and name of folders in (training, validation and test folders) in dataset folder  

# Vit-Base Model
num_layers     = 12                                                  # Number of transformer blocks
num_heads      = 12                                                  # Number of attention heads
hidden_dim     = 768                                                 # Dimension of the patch embeddings  
mlp_dim        = 3072                                                # Dimension of the MLP (feed-forward) layer


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path) 


def load_images(path, split=0.1):
    images = shuffle(glob(os.path.join(path, "*", "*.png")))
    return images                                                    # return images for x_train, x_val, and x_test 


def process_images(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (image_size, image_size))
    image = image/255.0

    #Pre-processing 
    patch_shape = (patch_size, patch_size, num_channels)
    patches = patchify(image, patch_shape, patch_size)
    patches = np.reshape(patches, flat_patches)
    patches = patches.astype(np.float32)

    # Labeling 
    class_name = path.split('\\')[-2]
    class_idx = class_names.index(class_name)
    class_idx = np.array(class_idx, dtype=np.int32)
    return patches, class_idx


def parse_data(path):
    patches, labels = tf.numpy_function(process_images, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, num_classes)
    patches.set_shape(flat_patches)
    labels.set_shape(num_classes)
    return patches, labels


def image_dataset(images, batch=patch_size):
    data = tf.data.Dataset.from_tensor_slices((images))
    data = data.map(parse_data)
    data = data.batch(batch)
    data = data.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    return data 


if __name__ == "__main__":
    
    create_dir("save_folder")
    
    train_path = "dataset\\training" 
    val_path = "dataset\\validation"
    
    model_path = os.path.join("save_folder", "ViT_model.h5")
    csv_path = os.path.join("save_folder", "multi_class_log.csv")

    x_train = load_images(train_path)
    x_val =   load_images(val_path)
    
    train_set = image_dataset(x_train, batch=batch_size)
    val_set =   image_dataset(x_val, batch=batch_size)
    
    model = ViT(num_patches, patch_size, num_channels, hidden_dim, num_layers, num_classes)
    model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(lr),metrics=["acc"])

    callback = [ ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1),
                  CSVLogger(csv_path),
                  EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)]

    model.fit(train_set, epochs=num_epochs, validation_data=val_set, callbacks=callback)
   
######################################### Training - end ################################################################   




######################################## Evaluation - start #############################################################

    test_path = "dataset\\test"
    model_path = os.path.join("save_folder", "ViT_model.h5")
    
    x_test = load_images(test_path)

    test_set = image_dataset(x_test, batch=batch_size)

    model.load_weights(model_path)

    model.evaluate(test_set)

######################################## Evaluation - end #############################################################

