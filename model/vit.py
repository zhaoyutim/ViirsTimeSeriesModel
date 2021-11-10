import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from model.vit.utilities.patches import Patches
from model.vit.vit_model import VisionTransformerGenerator
import matplotlib.pyplot as plt


# def visualizalize_patches():
#     plt.figure(figsize=(4, 4))
#     image = dataset_train[0][np.random.choice(range(dataset_train[0].shape[0]))]
#     plt.imshow(image.astype("uint8"))
#     plt.axis("off")
#
#     resized_image = tf.image.resize(
#         tf.convert_to_tensor([image]), size=(image_size, image_size)
#     )
#     patches = Patches(patch_size)(resized_image)
#     print(f"Image size: {image_size} X {image_size}")
#     print(f"Patch size: {patch_size} X {patch_size}")
#     print(f"Patches per image: {patches.shape[1]}")
#     print(f"Elements per patch: {patches.shape[-1]}")
#
#     n = int(np.sqrt(patches.shape[1]))
#     plt.figure(figsize=(4, 4))
#     for i, patch in enumerate(patches[0]):
#         ax = plt.subplot(n, n, i + 1)
#         patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
#         plt.imshow(patch_img.numpy().astype("uint8"))
#         plt.axis("off")

if __name__=='__main__':
    MAX_EPOCHS = 100
    dataset = np.load('../data/proj3_test.npy').transpose((1,0,2))
    print(dataset.shape)

    positive_sample = dataset[(dataset[:,:,45]>0).any(axis=1)]
    negative_sample = dataset[(dataset[:,:,45]==0).any(axis=1)]
    negative_sample = negative_sample[np.random.choice(negative_sample.shape[0], positive_sample.shape[0])]
    dataset = np.concatenate((positive_sample,negative_sample), axis=0)
    y_dataset = np.zeros((dataset.shape[0],dataset.shape[1],2))
    y_dataset[: ,:, 0] = dataset[:, :, 45] > 0
    y_dataset[:, :, 1] = dataset[:, :, 45] == 0

    x_train, x_test, y_train, y_test = train_test_split(dataset[:,:,:45], y_dataset, test_size=0.2)

    input_shape = x_train[0,:,:].shape
    # Patch parameters
    num_classes=2
    # image_size = 72  # We'll resize input images to this size
    # patch_size = 6  # Size of the patches to be extract from the input images
    # num_patches = (image_size // patch_size) ** 2

    # Transforer parameters
    projection_dim = 128
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8

    # Size of the dense layers of the final classifier
    mlp_head_units = [2048, 1024]

    # visualizalize_patches()

    vit_gen = VisionTransformerGenerator(input_shape, projection_dim, transformer_layers, num_heads, mlp_head_units, num_classes)

    history = vit_gen.run_experiment(x_train, y_train, x_test, y_test, batch_size=256, num_epochs=100, learning_rate=0.001, weight_decay=0.0001)