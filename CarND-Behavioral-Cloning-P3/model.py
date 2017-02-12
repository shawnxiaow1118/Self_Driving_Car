from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D

from keras.models import Model
from keras.layers import Input, merge
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import reader
import argparse


BATCH_SIZE = 256
DATA_DIR = 'all_log.csv'
DROP_PROB = 0.3
LEARNING_RATE = 1e-4
TEST_RATIO = 0.10
# nvidia model add a layer of 1x1 filters
def nvidia_model(dr=0.30):
    img_input = Input(shape=(66,200,3),name='image')
    conv1 = Convolution2D(3,1,1, border_mode='same',  activation = 'relu',subsample = (1,1),name='color_layer')(img_input)
    
    conv2 = Convolution2D(24,5,5, border_mode='valid', activation = 'relu', subsample = (2,2),name='conv2')(conv1)
#     conv2_drop = Dropout(dr)(conv2)
    conv3 = Convolution2D(36,5,5, border_mode='valid', activation = 'relu',subsample = (2,2),name='conv3')(conv2)
#     conv3_drop = Dropout(dr)(conv3)
    conv4 = Convolution2D(48,5,5, border_mode='valid', activation = 'relu', subsample = (2,2),name='conv4')(conv3)
#     conv4_drop = Dropout(dr)(conv4)
    conv5 = Convolution2D(64,3,3, border_mode='valid', activation = 'relu', subsample = (1,1),name='conv5')(conv4)
#     conv5_drop = Dropout(dr)(conv5)
    conv6 = Convolution2D(64,3,3, border_mode='valid', activation = 'relu', subsample = (1,1),name='conv6')(conv5)
    
    flat = Flatten()(conv6)
    flat_drop = Dropout(dr)(conv6)
    
    fc1 = Dense(100, activation = 'relu')(flat)
    fc1_drop = Dropout(dr)(fc1)
    fc2 = Dense(50, activation = 'relu')(fc1_drop)
    fc2_drop = Dropout(dr)(fc2)
    fc3 = Dense(10, activation = 'relu')(fc2_drop)
    predictions = Dense(1, activation = 'linear')(fc3)
    model = Model(input = [img_input], output=predictions)
    model.summary()
    return model


def get_arguments():
    parser = argparse.ArgumentParser(description='Nvidia net training')

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in train and valid batch.')
    parser.add_argument('--log', type=str, default=DATA_DIR,
                        help='The log file name')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Checkpoint file to restore model weights from.')

    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--drop_prob', type=float, default=DROP_PROB,
                        help='Dropout drop probability.')
    parser.add_argument('--test_ratio', type=float, default=TEST_RATIO,
                        help='Test dataset ratio.')
    parser.add_argument('--output', type=str, default="output.h5",
                        help='Model output.')
    return parser.parse_args()


def main():
    args = get_arguments()

    data = reader.data(args.log, args.batch_size, args.test_ratio)
    model = nvidia_model(args.drop_prob)

    model.compile(optimizer=Adam(args.learning_rate,decay=5e-5),
                  loss='mse')

    # save the model along training
    checkpoint = ModelCheckpoint('./best_model.hdf5', verbose=1, save_best_only=True)
    #early stopping
    early = EarlyStopping(monitor='val_loss',patience=2,verbose=1)

    # training
    history = model.fit_generator(data.next_train,samples_per_epoch=len(data.train_dataset), nb_epoch=10, verbose=1,
                       validation_data=data.next_valid, nb_val_samples=512,callbacks=[checkpoint, early])

    # save model
    model.save(args.output)
    print("Model saved to {}".format(args.output))

if __name__ == '__main__':
    main()