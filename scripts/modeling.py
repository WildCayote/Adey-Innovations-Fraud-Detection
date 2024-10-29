import mlflow, optuna, warnings, shutil, os
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from keras.api.layers import Input, LSTM, Conv1D, Dense, LSTM, Flatten, SimpleRNN

warnings.simplefilter(action='ignore')

class ModelingPipeline:
    """
    A class for containing methods that define different stages of modeling pipeline.
    """

    def __init__(self, x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, tracking_uri:str) -> None:
        """
        The class initializer. 

        Args:
            x_train(pd.DataFrame): the training set's features
            x_test(pd.DataFrame): the testing set's features
            y_train(pd.DataFrame): the training set's targets
            y_test(pd.DataFrame): the testing set's targets
            tracking_uri(str): the path to a folder or an sql file that serves as the tracking uri
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.tracking_uri = tracking_uri

        self.experiment_name = None
        self.experiment_id = None

        # add the tracking uri
        mlflow.set_tracking_uri(uri=self.tracking_uri)

    def initialize_mlflow(self, tracking_uri:str) -> None:
        """
        A method for defining the mlflow tracking uri

        Args:
            tracking_uri(str): the path to the tracking uri
        """

        # Set the tracking URI
        mlflow.set_tracking_uri(uri=tracking_uri)
    
    def create_experiment(self, experiment_name:str) -> str:
        """
        A method that will create an experiment in mlflow and return its id.
        If an experiment with the name already exists it will return the id of it.

        Args:
            experiment_name(str): the name of the experiment to be created

        Returns:
            experiment_id(str): the id of the experiment created.
        """

        self.experiment_name = experiment_name
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            search_result = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")
            experiment_id = search_result[0].experiment_id
        
        self.experiment_id = experiment_id
        return experiment_id

    def train_logistic_regressor(self) -> None:
        """
        A method that trains a logistic regressor for the dataset stored in the instance.
        """

        # create an experiment for hyperparameter tuning of the model
        experiment_id = mlflow.create_experiment(name="LogisticRegressor_Tuning")

        # define the objective function for optuna
        def objective(trial):
            with mlflow.start_run(nested=True, run_name="LogisticRegressor", experiment_id=experiment_id):
                C = trial.suggest_loguniform('C', 1e-5, 1e2)
                max_iter = trial.suggest_int('max_iter', 50, 500)

                # Create and train the logistic regression model
                model = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear', random_state=7)
                model.fit(self.x_train, self.y_train)

                # Evaluate the model and log the accuracy
                accuracy = model.score(self.x_test, self.y_test)

                # Log metrics and hyperparameters with MLflow
                mlflow.log_params({'C': C, 'max_iter': max_iter})
                mlflow.log_metric('accuracy', accuracy)

            return accuracy

        # create an optuna study
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="LogisticRegressor"):
            study = optuna.create_study(direction='maximize')

            # Run the hyper-parameter optimization
            study.optimize(objective, n_trials=7)
    
            # Use the best parameters to train the final model
            best_params = study.best_params
            final_model = LogisticRegression(**best_params, solver='liblinear', random_state=7)
            final_model.fit(self.x_train, self.y_train)
    
            # Log the model with MLflow
            mlflow.log_params(best_params)
            mlflow.sklearn.log_model(final_model, "best_logistic_regressor")
    
            # Log final model performance on the test set
            predictions = final_model.predict(X=self.x_test)
            accuracy = accuracy_score(y_true=self.y_test, y_pred=predictions)
            precision = precision_score(y_true=self.y_test, y_pred=predictions)
            f1 = f1_score(y_true=self.y_test, y_pred=predictions)
            recall = recall_score(y_true=self.y_test, y_pred=predictions)
            
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1_score', f1)
            mlflow.log_metric('recall', recall)

            print(f"#### Finished Training Logistic Regressor ####")
            print(f"Best parameters for Logistic Regressor: {best_params}")
            print(f"Test accuracy with best parameters: {accuracy}")

    def train_decision_tree(self) -> None:
        """
        A method that trains a decision tree for the dataset stored in the instance.
        """
        experiment_id = mlflow.create_experiment(name="DecisionTree_Tuning")
    
        def objective(trial):
            with mlflow.start_run(nested=True, run_name="DecisionTree", experiment_id=experiment_id):
                max_depth = trial.suggest_int('max_depth', 2, 32)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=7)
                model.fit(self.x_train, self.y_train)
    
                accuracy = model.score(self.x_test, self.y_test)
                mlflow.log_params({'max_depth': max_depth, 'min_samples_split': min_samples_split})
                mlflow.log_metric('accuracy', accuracy)
    
            return accuracy
    
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="DecisionTree"):
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=7)
    
            best_params = study.best_params
            final_model = DecisionTreeClassifier(**best_params, random_state=7)
            final_model.fit(self.x_train, self.y_train)
    
            mlflow.log_params(best_params)
            mlflow.sklearn.log_model(final_model, "best_decision_tree")
    
            predictions = final_model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)
            recall = recall_score(self.y_test, predictions)
    
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1_score', f1)
            mlflow.log_metric('recall', recall)
    
            print(f"#### Finished Training Decision Tree ####")
            print(f"Best parameters for Decision Tree: {best_params}")
            print(f"Test accuracy with best parameters: {accuracy}")
    
    def train_random_forest(self) -> None:
        """
        A method that trains a random forest for the dataset stored in the instance.
        """
        experiment_id = mlflow.create_experiment(name="RandomForest_Tuning")
    
        def objective(trial):
            with mlflow.start_run(nested=True, run_name="RandomForest", experiment_id=experiment_id):
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                max_depth = trial.suggest_int('max_depth', 2, 32)
    
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=7)
                model.fit(self.x_train, self.y_train)
    
                accuracy = model.score(self.x_test, self.y_test)
                mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
                mlflow.log_metric('accuracy', accuracy)
    
            return accuracy
    
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="RandomForest"):
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=7)
    
            best_params = study.best_params
            final_model = RandomForestClassifier(**best_params, random_state=7)
            final_model.fit(self.x_train, self.y_train)
    
            mlflow.log_params(best_params)
            mlflow.sklearn.log_model(final_model, "best_random_forest")
    
            predictions = final_model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)
            recall = recall_score(self.y_test, predictions)
    
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1_score', f1)
            mlflow.log_metric('recall', recall)
    
            print(f"#### Finished Training Random Forest ####")
            print(f"Best parameters for Random Forest: {best_params}")
            print(f"Test accuracy with best parameters: {accuracy}")
     
    def train_mlp(self) -> None:
        """
        A method that trains a multi-layer perceptron for the dataset stored in the instance.
        """
        experiment_id = mlflow.create_experiment(name="MLP_Tuning")
    
        def objective(trial):
            with mlflow.start_run(nested=True, run_name="MLP", experiment_id=experiment_id):
                hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 50, 200)
                alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
                max_iter = trial.suggest_int('max_iter', 50, 500)
    
                model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), alpha=alpha, max_iter=max_iter, random_state=7)
                model.fit(self.x_train, self.y_train)
    
                accuracy = model.score(self.x_test, self.y_test)
                mlflow.log_params({'hidden_layer_sizes': hidden_layer_sizes, 'alpha': alpha, 'max_iter': max_iter})
                mlflow.log_metric('accuracy', accuracy)
    
            return accuracy
    
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="MLP"):
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=7)
    
            best_params = study.best_params
            final_model = MLPClassifier(hidden_layer_sizes=(best_params['hidden_layer_sizes'],), alpha=best_params['alpha'], max_iter=best_params['max_iter'], random_state=7)
            final_model.fit(self.x_train, self.y_train)
    
            mlflow.log_params(best_params)
            mlflow.sklearn.log_model(final_model, "best_mlp")
    
            predictions = final_model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)
            recall = recall_score(self.y_test, predictions)
    
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1_score', f1)
            mlflow.log_metric('recall', recall)
    
            print(f"#### Finished Training MLP ####")
            print(f"Best parameters for MLP: {best_params}")
            print(f"Test accuracy with best parameters: {accuracy}")

    def train_cnn(self) -> None:
        """
        A method that trains a Convolutional Neural Network (CNN) for the dataset stored in the instance.
        """
        
        mlflow.tensorflow.autolog()
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="CNN"):
            model = models.Sequential()
            model.add(Input(shape=(self.x_train.shape[1], 1)))
            model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(self.x_train, self.y_train, epochs=10, validation_data=(self.x_test, self.y_test),
                      callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

            predictions = (model.predict(self.x_test) > 0.5).astype("int32").flatten()
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)
            recall = recall_score(self.y_test, predictions)

            mlflow.log_metrics({'accuracy': accuracy, 'precision': precision, 'f1_score': f1, 'recall': recall})
            mlflow.tensorflow.log_model(model, "best_cnn")

            print(f"#### Finished Training CNN ####")
            print(f"Test accuracy: {accuracy}")

    def train_rnn(self) -> None:
        """
        A method that trains a simple Recurrent Neural Network (RNN) for the dataset stored in the instance.
        """
        mlflow.tensorflow.autolog()
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="RNN"):
            model = models.Sequential()
            model.add(Input(shape=(self.x_train.shape[1], 1)))
            model.add(SimpleRNN(units=32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(
                self.x_train, 
                self.y_train, 
                epochs=10, 
                validation_data=(self.x_test, self.y_test),
                callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
            )

            predictions = (model.predict(self.x_test) > 0.5).astype("int32").flatten()
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)
            recall = recall_score(self.y_test, predictions)

            mlflow.log_metrics({
                'accuracy': accuracy, 
                'precision': precision, 
                'f1_score': f1, 
                'recall': recall
            })
            mlflow.tensorflow.log_model(model, "best_rnn")

            print("#### Finished Training RNN ####")
            print(f"Test accuracy: {accuracy}")

    def train_lstm(self) -> None:
        """
        A method that trains a Long Short-Term Memory (LSTM) network for the dataset stored in the instance.
        """

        mlflow.tensorflow.autolog()
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="LSTM"):
            model = models.Sequential()
            model.add(Input(shape=(self.x_train.shape[1], 1)))
            model.add(LSTM(50))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(self.x_train, self.y_train, epochs=7, validation_data=(self.x_test, self.y_test),
                      callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

            predictions = (model.predict(self.x_test) > 0.5).astype("int32").flatten()
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)
            recall = recall_score(self.y_test, predictions)

            mlflow.log_metrics({'accuracy': accuracy, 'precision': precision, 'f1_score': f1, 'recall': recall})
            mlflow.tensorflow.log_model(model, "best_lstm")

            print(f"#### Finished Training LSTM ####")
            print(f"Test accuracy: {accuracy}")

    def log_best_model(self, export_path:str) -> None:
        """
        A method that will search for the best performing model from the experiment and then save it to a specified path.

        Args:
            export_path(str): the path where you want the best model to be moved/copied to
        """

    def train_models(self) -> None:
        """
        A method to trigger training for all of the models present.

        **Note:** You can add more methods to train specific models then add the method inside to be included with all of the other models.
        """
        self.train_logistic_regressor()
        self.train_decision_tree()
        self.train_random_forest()
        self.train_mlp()
        self.train_cnn()
        self.train_rnn()
        self.train_lstm()

        print("#### Finished Training All Models ####")
