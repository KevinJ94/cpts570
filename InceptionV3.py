from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam, SGD, RMSprop
import Helper as hp
import clr_callback


def demo_tf(opt, epochs):
    BREED_NUM = 120
    EPOCHS_NUM = epochs
    train_generator, validation_generator = hp.image_process()

    # Get the InceptionV3 model so we can do transfer learning
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(400, 400, 3))
    # base_model = InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape=(DIMENSIONS, DIMENSIONS, 3))

    # Add a global spatial average pooling layer
    x = base_model.output
    # x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    # Add a fully-connected layer and a logistic layer with X breeds (classes)
    # (there will  120 breeds (classes) for the final submission)
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.3)(x)
    predictions = Dense(BREED_NUM, activation='softmax')(x)

    # The model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
    # print(len(base_model.layers))
    for layer in base_model.layers:
        layer.trainable = False

    # possible to experiment with SGD?
    # Compile with Adam optimisation, learning rate = 0.0001
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # clr = clr_callback.CyclicLR(base_lr=0.001, max_lr=0.009,
    #                            step_size=128)
    #
    # Train the model
    result = model.fit_generator(train_generator,
                                 steps_per_epoch=32,
                                 validation_data=validation_generator,
                                 validation_steps=32,
                                 epochs=EPOCHS_NUM,
                                 #                            callbacks=[clr],
                                 verbose=2)
    model.save("InceptionV3_training_accuracy_" + str(EPOCHS_NUM) + '.h5')
    y_axis = result.history["accuracy"]
    x_axis = list(range(1, len(y_axis) + 1))
    hp.save_curve(x_axis=x_axis, y_axis=y_axis, title="InceptionV3_training_accuracy_" + str(EPOCHS_NUM),
                  xlabel="epoch",
                  ylabel="accuracy")
    y_axis = result.history["val_accuracy"]
    hp.save_curve(x_axis=x_axis, y_axis=y_axis, title="InceptionV3_testing_accuracy_" + str(EPOCHS_NUM), xlabel="epoch",
                  ylabel="accuracy")


demo_tf(RMSprop(), epochs=50)
