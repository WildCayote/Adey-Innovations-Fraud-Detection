import math, warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore")

sns.set_theme()

class EDAAnalyzer:
    """
    A class for organizing functions/methods for performing EDA on bank transaction data.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the EDAAnalyzer class

        Args:
            data(pd.DataFrame): the dataframe that contains bank transactional data
        """
        self.data = data
    
    def basic_overview(self) -> None:
        """
        A function that creates basic overview of the data like - data type of columns, the shape of the data(i.e the number of rows and columns) 
        """
        # print out the shape
        print(f"The data has a shape of: {self.data.shape}")

        # print out the column info
        print(self.data.info())
    
    def summary_statistics(self) -> None:
        """
        A function that generates 5 number summary(descriptive statistics) of the dataframe
        """
        print(self.data.describe())
    
    def missing_values(self) -> None:
        """
        A function that checks for columns with missing value and then returns ones with greater than 0 with the percentage of missing values.
        """

        # obtain missing value percentage
        missing = self.data.isna().mean() * 100
        missing = missing [missing > 0]
        
        # print out the result
        print(f"These are columns with missing values greater than 0%:\n{missing}")

    def determine_duplicate(self) -> None:
        """
        A function that finds the percentage of duplicate data/rows in the given data set. It prints the percentage of that duplicate data
        """

        # obtain ratio of dublicate data
        duplicate_ratio = self.data.duplicated().mean()

        # calcualte the percentage from ration
        percentage = duplicate_ratio * 100

        # print out the result
        print(f"The data has {percentage}% duplicate data.")

    def numerical_distribution(self) -> None:
        """
        A function that will give histogram plots of numerical data with a density curve that shows the distribution of data
        """
        
        # determine the numerical columns and data
        numerical_data = self.data._get_numeric_data()
        numerical_cols = numerical_data.columns

        # detrmine number of rows and columns for 
        num_cols = math.ceil(len(numerical_cols) ** 0.5)

        # calculate the number of rows
        num_rows = math.ceil(len(numerical_cols) / num_cols)

        # create subpltos
        fig, axes = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=(14, 9))

        # flatten the axes
        axes = axes.flatten()

        for idx, column in enumerate(numerical_cols):
            # calculate the median and mean to use in the plots
            median = self.data[column].median()
            mean = self.data[column].mean()

            # plot the histplot for that column with a density curve overlayed on it
            sns.histplot(self.data[column], bins=15, kde=True, ax=axes[idx])

            # add title for the subplot
            axes[idx].set_title(f"Distribution plot of {column}", fontsize=10)

            # set the x and y labels
            axes[idx].set_xlabel(column, fontsize=9)
            axes[idx].set_ylabel("Frequency", fontsize=9)

            # add a lines for indicating the mean and median for the distribution
            axes[idx].axvline(mean, color='black', linewidth=1, label='Mean') # the line to indicate the mean
            axes[idx].axvline(median, color='red', linewidth=1, label='Median') # the line to indivate the median 

            # add legends for the mean and median
            axes[idx].legend()

        # remove unused subplots
        for unused in range(idx + 1, len(axes)):
            plt.delaxes(ax=axes[unused])
        
        # create a tight layout
        plt.tight_layout()

        # show the plot
        plt.show()

    def categorical_distribution(self) -> None:
        """
        A function that will give bar plots of categorical data
        """
        # define the categorical columns that are worth investigating, 
        # i.e avoid columns which hold ids, timestamps because the information they provide when looking at their distribution isn't useful
        columns_of_interest = ["source", "browser", "sex"]

        # create subplots for each categorical variable
        num_columns = math.ceil(len(columns_of_interest) ** 0.5)
        num_rows = math.ceil(len(columns_of_interest) / num_columns) 

        fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=(14,9))

        axes = axes.flatten()

        for axis_idx, column in enumerate(columns_of_interest):
            # group the data around that column and count instances with respective values
            column_grouping = self.data.groupby(by=column)
            grouping_counts = column_grouping.size().sort_values()

            # create the bar plot
            sns_plot = sns.barplot(data=grouping_counts, ax=axes[axis_idx], palette='husl')
            sns_plot.tick_params(axis='x', labelrotation=45)
            sns_plot.set_ylabel(ylabel="Count", weight='bold')
            sns_plot.set_xlabel(xlabel=column, weight='bold', loc='center', labelpad=5)

            category_values = grouping_counts.keys()
            for idx, patch in enumerate(sns_plot.patches):
                # get the corrdinates to write the values 
                x_coordinate = patch.get_x() + patch.get_width() / 2
                y_coordinate = patch.get_height()

                # get the value to be written
                value = grouping_counts[category_values[idx]]
                sns_plot.text(x=x_coordinate, y=y_coordinate, s=value, ha='center', va='bottom', weight='bold')

        # remove unused subplots
        for unused in range(axis_idx + 1, len(axes)):
            plt.delaxes(ax=axes[unused])    

        # create a tight layout
        plt.tight_layout(pad=4)

        # show the plot
        plt.show()

    def correlation_analysis(self) -> None:
        """
        A function that performs correlation analysis by creating heatmap plots between the numerical variabe;s
        """
        # calculate the correlation matrix
        correlation_matrix = self.data._get_numeric_data().corr()

        # plot it as a heatmap using seaborn
        cmap = sns.color_palette("crest", as_cmap=True)
        ax = sns.heatmap(correlation_matrix, cmap=cmap, annot=True)
        ax.set_title("Correlation Matrix Heatmap", weight='bold', fontsize=20, pad=20)

    def pair_plot(self) -> None:
        """
        A function that creates a pair plot between specified numerical columns
        """

        # define the numerical columns
        numerical_cols = ['purchase_value', 'age', 'class']

        sns.pairplot(self.data[numerical_cols], palette='husl', hue='class')
        plt.show()

    def outlire_detection(self) -> None:
        """
        A function that performs outlire detection by plotting a box plot.
        """
        # create the box plots of the numeric data
        ax = sns.boxplot(data=self.data, palette='husl')
        ax.set_title("Box-plot of Categorical Variables", pad=30, fontweight='bold')
        ax.set_xlabel("Numerical Columns", fontweight='bold', labelpad=10)
        ax.set_ylabel("Values", fontweight='bold', labelpad=10)

    def count_outliers(self) -> None:
        """
        A function that counts the number of outliers in numerical columns. The amount of data that are outliers and also gives the cut-off point.
        The cut off points being defined as:
            - lowerbound = Q1 - 1.5 * IQR
            - upperbound = Q3 + 1.5 * IQR
        """
        # get the numeric data
        numerical_columns = list(self.data._get_numeric_data().columns)
        numerical_data = self.data[numerical_columns]

        # obtain the Q1, Q3 and IQR(Inter-Quartile Range)
        quartile_one = numerical_data.quantile(0.25)
        quartile_three = numerical_data.quantile(0.75)
        iqr = quartile_three - quartile_one

        # obtain the upperbound and lowerbound values for each column
        upper_bound = quartile_three + 1.5 * iqr
        lower_bound = quartile_one - 1.5 * iqr

        # count all the outliers for the respective columns
        outliers = {"Columns" : [], "Num. of Outliers": []}
        for column in lower_bound.keys():
            column_outliers = self.data[(self.data[column] < lower_bound[column]) | (self.data[column] > upper_bound[column])]
            count = column_outliers.shape[0]

            outliers["Columns"].append(column)
            outliers["Num. of Outliers"].append(count)

        outliers = pd.DataFrame.from_dict(outliers).sort_values(by='Num. of Outliers')
        ax = sns.barplot(outliers, x='Columns', y='Num. of Outliers', palette='husl')
        ax.set_title("Plot of Skewness values of Numerical Columns", pad=20)
        ax.set_xlabel("Numerical Columns", weight='bold')
        ax.set_ylabel("Num. of Outliers", weight="bold")
        ax.tick_params(axis='x', labelrotation=45)

        columns = outliers['Columns'].unique()
        for idx, patch in enumerate(ax.patches):
            # get the corrdinates to write the values
            x_coordinate = patch.get_x() + patch.get_width() / 2
            y_coordinate = patch.get_height()

            # get the value of the coordinate
            value = outliers[outliers['Columns'] == columns[idx]]['Num. of Outliers'].values[0]
            ax.text(x=x_coordinate, y=y_coordinate, s=value, ha='center', va='bottom', weight='bold')

    def investigate_country(self, ip_mapping: pd.DataFrame) -> None:
        """
        A function that will try to find trends of frauds for each country and plot the top 5 countries with the most frauds.

        Args:
            ip_mapping(pd.DataFrame): a dataframe that contains the ip mapping for each country
        """

        # convert the ip_address into integers
        self.data['ip_address'] = self.data['ip_address'].astype(dtype=int)
        self.data.sort_values('ip_address', inplace=True)

        # convert the ranges into integers
        ip_mapping[['lower_bound_ip_address', 'upper_bound_ip_address']] = ip_mapping[['lower_bound_ip_address', 'upper_bound_ip_address']].astype(dtype=int)

        # sort the lowerbound_ips
        ip_mapping.sort_values('lower_bound_ip_address', inplace=True)

        # merge the data
        merged = pd.merge_asof(self.data, ip_mapping, left_on='ip_address', right_on='lower_bound_ip_address', direction='nearest')
        self.data = merged.drop(columns=['upper_bound_ip_address', 'lower_bound_ip_address'])

        # Convert purchase_time to datetime format
        self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])

        # Group fraudulent transactions by day and country
        daily_fraud = self.data[self.data['class'] == 1].groupby([self.data['purchase_time'].dt.date, 'country']).size().unstack().fillna(0)

        # Determine the top countries with the highest number of fraudulent transactions
        top_countries = daily_fraud.sum().nlargest(5).index

        # Plotting fraud trends over time for the top countries
        plt.figure(figsize=(14, 7))
        daily_fraud[top_countries].plot()
        plt.title('Daily Fraudulent Transactions for The Top 5 Countries')
        plt.xlabel('Date')
        plt.ylabel('Number of Fraudulent Transactions')
        plt.legend(title='Country')
        plt.show()

