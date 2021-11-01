
import numpy as np
from sklearn.model_selection import train_test_split

from model.gru.gru_model import GRUModel

if __name__ == '__main__':
    MAX_EPOCHS = 100
    dataset = np.load('../data/proj3_training.npy').transpose((1,0,2))
    print(dataset.shape)
    y_dataset = np.zeros((dataset.shape[0],dataset.shape[1],2))
    y_dataset[: ,:, 0] = dataset[:, :, 45] > 0
    y_dataset[:, :, 1] = dataset[:, :, 45] == 0
    x_train, x_test, y_train, y_test = train_test_split(dataset[:,:,:45], y_dataset, test_size=0.2)
    gru = GRUModel()
    # TODO: Normalization only on train data
    performance = {}
    history = gru.compile_and_fit(gru.model, x_train, y_train)

    performance['LSTM'] = gru.model.evaluate(x_test, y_test, verbose=0)