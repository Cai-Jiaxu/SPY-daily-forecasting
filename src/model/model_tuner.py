import tensorflow as tf
import keras_tuner as kt

class ModelTuner:
    def __init__(self, train_df, val_df, test_df):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tuner = None

    def model_builder(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=32, return_sequences=True))
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(tf.keras.layers.LSTM(units=hp_units, return_sequences=True))
        model.add(tf.keras.layers.Dense(1))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

        return model

    def perform_hyperparameter_tuning(self):
        self.tuner = kt.Hyperband(self.model_builder,
                             objective='val_loss',
                             max_epochs=20,
                             factor=3,
                             directory='hyperband_dir',
                             project_name='lstm_hyperband')

        # Define a callback to stop tuning when the validation loss stops improving
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Perform hyperparameter tuning
        self.tuner.search(self.train_df, epochs=20, validation_data=self.val_df, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        return best_hps

    def show_hp(self, best_hps):
        best_units = best_hps.get('units')
        best_learning_rate = best_hps.get('learning_rate')

        print("Best Units:", best_units)
        print("Best Learning Rate:", best_learning_rate)

    def build_train_evaluate_model(self, best_hps):
        # Build the final LSTM model with the best hyperparameters
        final_model = self.tuner.hypermodel.build(best_hps)

        # Train the final model
        final_model.fit(self.train_df, epochs=20, validation_data=self.val_df)

        return final_model

    def show_loss(self, final_model):
        # Evaluate the final model
        loss = final_model.evaluate(self.test_df)
        print("Test Loss:", loss)