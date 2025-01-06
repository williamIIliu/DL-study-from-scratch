import torch
import torch.nn as nn
def VGG16_1d(classes = 3):
    img_input = Input((999,13))
    # Block 1
    x = nn.Conv1d(64, 3,
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = nn.Conv1d(64, 3,
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = nn.MaxPooling1D(2, strides=2, name='block1_pool', padding='same')(x)

    # Block 2
    x = nn.Conv1d(128, 3,
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = nn.Conv1d(128, 3,
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = nn.MaxPooling1D(2, strides=2, name='block2_pool', padding='same')(x)

    # Block 3
    x = nn.Conv1d(256, 3,
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = nn.Conv1d(256, 3,
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = nn.Conv1d(256, 3,
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = nn.MaxPooling1D(2, strides=2, name='block3_pool', padding='same')(x)

    # Block 4
    x = nn.Conv1d(512, 3,
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = nn.Conv1d(512, 3,
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = nn.Conv1d(512, 3,
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = nn.MaxPooling1D(2, strides=2, name='block4_pool', padding='same')(x)

    # Block 5
    x = nn.Conv1d(512, 3,
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = nn.Conv1d(512, 3,
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = nn.Conv1d(512, 3,
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = nn.MaxPooling1D(2, strides=2, name='block5_pool', padding='same')(x)

    # Classification block
    x = nn.Flatten(name='flatten')(x)
    x = nn.Dense(128, activation='relu', name='fc1')(x) # reduced dim for 1-d task
    x = nn.Dense(128, activation='relu', name='fc2')(x)
    x = nn.Dense(classes, activation='softmax', name='predictions')(x)


    # Create model.
    model = models.Model(img_input, x, name='vgg16')
    return model

model = VGG16_1d(3)
model.summary()