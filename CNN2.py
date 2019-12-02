from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import Helper as hp



training_set, test_set = hp.image_process()
epochs_num = 50

model = Sequential()

# -----------------------------------------------------------------------------------
# conv 1
model.add(Conv2D(16, (3,3), input_shape=(150, 150, 3)))       # input -N,150,150,3, output- N,148,148,16
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# max pool 1
model.add(MaxPooling2D(pool_size=(2,2),strides=2))                                   #input- N,148,148,16, output- N, 74,74,16

# -----------------------------------------------------------------------------------
# # conv 2
model.add(Conv2D(32, (3,3)))                                                         #input- N,74,74,16 output - N, 72,72,16
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# max pool 2
model.add(MaxPooling2D(pool_size=(2,2),strides=2))                                 #input - N,72,72,16, output- N,36,36,16
# -----------------------------------------------------------------------------------

# conv 3
model.add(Conv2D(64, (3,3)))                                                       #input - N,36,36,16, output- N,34,34,32
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# max pool 3
model.add(MaxPooling2D(pool_size=(2,2),strides=2))                                #input- N,34,34,32, output- N,17,17,32
# -----------------------------------------------------------------------------------

# # conv 4
model.add(Conv2D(128, (3,3)))                                                     #input- N,17,17,32, output- N,15,15,32
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Dropout(0.7))
# max pool 4
model.add(MaxPooling2D(pool_size=(2,2),strides=2))                              #input- N,15,15,32, output- N,7,7,32

# flatten
model.add(Flatten())                                                            # output- 1568

# fc layer 1
# model.add(Dense(1024, activation='relu'))

# fc layer 2
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
# fc layer 3
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# fc layer 4
model.add(Dense(120, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
result = model.fit_generator(
    training_set,
    epochs=epochs_num,
    validation_data=test_set,
)
#model.save('epochs_' + str(epochs_num) + '.h5')
y_axis = result.history["accuracy"]
x_axis = list(range(1, len(y_axis) + 1))
hp.save_curve(x_axis=x_axis, y_axis=y_axis, title="CNN2_training_accuracy_with_3_layers_"+ str(epochs_num), xlabel="epoch",
              ylabel="accuracy")
y_axis = result.history["val_accuracy"]
hp.save_curve(x_axis=x_axis, y_axis=y_axis, title="CNN2_testing_accuracy_with_3_layers_"+str(epochs_num), xlabel="epoch",
              ylabel="accuracy")