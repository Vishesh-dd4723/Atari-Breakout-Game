from models import Model1
from helper_funcs import *
from tensorflow.keras import losses, optimizers, metrics
from sklearn.model_selection import train_test_split

path = '/'
data_file_name = 'data.csv'
data = pd.read_csv(path + data_file_name)
trained_model_name = 'model3'

model_ = Model1()
model = model_.model((80, 75, 3), 4)
# print(model.summary())
lr = 0.003
model.compile(optimizer=optimizers.RMSprop(learning_rate=lr), loss=losses.MeanAbsoluteError(), metrics=metrics.MeanAbsoluteError())

Y = data['reward'].to_numpy()
actions = data['action'].to_numpy()
X = data.drop(columns=['reward', 'action']).to_numpy()
X = merge_S_A(X, actions)
X = np.reshape(X, newshape=((X.shape[0], 80, 75, 3)))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)
# X_train.shape
model.fit(X_train, Y_train, epochs=8, batch_size=64)

# model.evaluate(X_test, Y_test)
model.save(path + trained_model_name)