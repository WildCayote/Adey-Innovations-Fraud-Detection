import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular 
from utils import load_mlflow_model

class ModelExplainer:
    """
    A class which uses SHAP and LIME to explain the decision of models.
    """

    def __init__(self, X:pd.DataFrame, model_path: str):
        """
        Creates an instance of the class.

        Args:
            X(pd.DataFrame): the input features we want to perform explanation on
            model_path(str): the path to the folder that contains the mlflow model
        """

        self.features = X
        self.model = load_mlflow_model(parent_folder=model_path)
    
    def shap_explanations(self, sample_index:int = 0, background_sample_size:int = 100) -> None:
        """
        Perform shap model explanations such as: summary plot, force plot and dependence plot

        Args:
            sample_index(int): the index of the example/data we want to plot the force plot
            background_sample_size (int): Number of background samples to use in KernelExplainer.
        """
        # Sample the background data to improve performance
        background_data = shap.sample(self.features, background_sample_size, random_state=7)

        # initiate shap explainer and obtain shap valeus
        explainer = shap.KernelExplainer(model=self.model.predict, data=background_data)
        shap_values = explainer.shap_values(X=background_data)

        # Reshape shap_values to handle the extra dimension if necessary
        if shap_values.ndim == 3:
            shap_values = np.squeeze(shap_values)

        # print the obtained shap values
        print(f"Types of the SHAP values: {type(shap_values)}")
        print(f"Shape of the SHAP values: {shap_values.shape}")

        # create summary plot
        plt.figure(figsize=(15, 4))
        shap.summary_plot(shap_values, background_data, show=False)
        plt.title('Summary Plot')
        plt.show()

        # Plot force plot for the selected sample instance from the data
        shap.plots.force(explainer.expected_value, shap_values[sample_index],feature_names=self.features.columns)
       
       # SHAP Dependence Plot: Relationship between feature and model output
        shap.dependence_plot(background_data.columns[0], shap_values, background_data, show=False)
        plt.title(f'Dependence Plot for Feature: {background_data.columns[0]}')
        plt.show()

    def lime_explanations(self, sample_index: int = 1) -> None:
        """
        Perform LIME model explanations for feature importance.
    
        Args:
            sample_index(int): Index of the data example for which to generate the LIME explanation.
        """
        
        # Initialize the LIME explainer with the background data
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.features),
            feature_names=self.features.columns,
            mode='regression'
        )
        
        # Select an instance from the data to explain
        instance = self.features.iloc[sample_index]
    
        # Generate LIME explanation for the selected instance
        explanation = explainer.explain_instance(instance, self.model.predict)
    
        # Plot feature importance using LIME
        plt.figure(figsize=(10, 6))
        explanation.as_pyplot_figure()
        plt.title(f'LIME Explanation for Instance {sample_index}')
        plt.show()

     

