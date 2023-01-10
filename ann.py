from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback


from timeit import default_timer as timer

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

def compile_model(model):
    # output the model's structure
    model.summary()

    # Model compilation
    optimizer = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    time_cb = TimingCallback()
    callbacks = [time_cb]

    # Model Training
    history = model.fit(x_train, y_onehot_train, epochs=10, batch_size=100, validation_data=(x_test, y_onehot_test), callbacks=callbacks)

    return history, sum(time_cb.logs)


model = Sequential()
model.add(Dense(1000, activation='relu', input_dim = 784))
model.add(Dense(1000, activation='relu'))
model.add(Dense(10,activation='softmax'))



history, time_spent = compile_model(model)
view_model_results(model, history)
print('Time spent', time_spent)
