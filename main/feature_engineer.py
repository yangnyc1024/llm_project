import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.model_selection import train_test_split

class DataFrameSplitter:
    def __init__(self, df: pd.DataFrame, train_size: int):
        """
        Initializes the DataFrameSplitter with the input DataFrame and train size.

        Args:
        - df: The input DataFrame containing sentiment data.
        - train_size: The number of samples per sentiment category for the train and test sets.
        """
        self.df = df
        self.train_size = train_size

    def stratified_split(self) -> tuple:
        """
        Performs a stratified split of the DataFrame into train and test sets based on sentiment categories.

        Returns:
        - A tuple of DataFrames: (X_train, X_test)
        """
        X_train, X_test = [], []
        
        # Perform stratified split for each sentiment category
        for sentiment in ["positive", "neutral", "negative"]:
            train, test = train_test_split(self.df[self.df.sentiment == sentiment], 
                                           train_size=self.train_size, 
                                           test_size=self.train_size, 
                                           random_state=42)
            X_train.append(train)
            X_test.append(test)

        # Shuffle the train data and reset index
        X_train = pd.concat(X_train).sample(frac=1, random_state=10).reset_index(drop=True)
        X_test = pd.concat(X_test).reset_index(drop=True)
        
        return X_train, X_test

    def create_eval_set(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Creates an evaluation set with the remaining data that is not in either the train or test sets.

        Args:
        - X_train: The training DataFrame.
        - X_test: The testing DataFrame.

        Returns:
        - X_eval: The evaluation DataFrame.
        """
        # eval_idx = [idx for idx in self.df.index if idx not in X_train.index and idx not in X_test.index]
        eval_idx = [idx for idx in self.df.index if idx not in list(X_train.index) + list(X_test.index)]

        # X_eval = self.df.loc[eval_idx]
        X_eval = self.df[self.df.index.isin(eval_idx)]


        # Create a balanced evaluation set with 50 samples per sentiment category
        # X_eval = (X_eval.groupby('sentiment', group_keys=False)
        #                  .apply(lambda x: x.sample(n=50, random_state=10, replace=True))
        #                  .reset_index(drop=True))
        
        X_eval = (X_eval
          .groupby('sentiment', group_keys=False)
          .apply(lambda x: x.sample(n=50, random_state=10, replace=True)))
    
        X_train = X_train.reset_index(drop=True)
        
        return X_eval

    def split(self) -> tuple:
        """
        Orchestrates the splitting process to generate train, test, and eval DataFrames.

        Returns:
        - A tuple of DataFrames: (X_train, X_test, X_eval)
        """
        X_train, X_test = self.stratified_split()
        X_eval = self.create_eval_set(X_train, X_test)

        return X_train, X_test, X_eval



# # Assuming 'df' is your DataFrame containing sentiment data
# splitter = DataFrameSplitter(df, train_size=300)

# # Perform the split to obtain train, test, and eval sets
# X_train_df, X_test_df, X_eval_df = splitter.split()

# # Print shapes and target column breakup of the train data
# print(f'Shape of train, test, and eval data are {X_train_df.shape}, {X_test_df.shape}, {X_eval_df.shape}')
# print(f'\nBreak up by target column in train data is\n{X_train_df.sentiment.value_counts()}')


