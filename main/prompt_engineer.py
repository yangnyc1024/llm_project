import pandas as pd

class PromptGenerator:
    def __init__(self):
        """
        Initializes the PromptGenerator with prompt templates for training/validation and testing.
        """
        self.prompt_template = """
        Analyze the sentiment of the news headline enclosed in square brackets, 
        determine if it is positive, neutral, or negative, and return the answer as 
        the corresponding sentiment label "positive", "neutral", or "negative".

        [{}] = {}""".strip()

        # self.test_prompt_template = """
        # For the given news headline in square brackets, predict its sentiment as either 
        # "positive", "neutral", or "negative". Do not consider the provided sentiment label.

        # [{}] = ?
        # """.strip()
        self.test_prompt_template = """
        Analyze the sentiment of the news headline enclosed in square brackets, 
        determine if it is positive, neutral, or negative, and return the answer as 
        the corresponding sentiment label "positive" or "neutral" or "negative".

        [{}] = """.strip()


    def generate_prompt(self, data_point: pd.Series) -> str:
        """
        Generates a prompt for sentiment analysis based on the provided data point.

        Args:
        - data_point: A Series object containing the text and sentiment of a news headline.

        Returns:
        - A formatted string prompt for sentiment analysis.
        """
        return self.prompt_template.format(data_point["text"], data_point["sentiment"])

    def generate_test_prompt(self, data_point: pd.Series) -> str:
        """
        Generates a test prompt for sentiment prediction based on the provided news headline.

        Args:
        - data_point: A Series object containing the text of a news headline.

        Returns:
        - A formatted string prompt for sentiment prediction without revealing the actual sentiment.
        """
        return self.test_prompt_template.format(data_point["text"])

    def generate_dataframe_prompts(self, df, prompt_type='train'):
        """
        Generates prompts for an entire DataFrame.

        Args:
        - df: DataFrame containing the text (and sentiment for training) of news headlines.
        - prompt_type: Type of prompts to generate ('train' for training/validation prompts, 'test' for test prompts).

        Returns:
        - DataFrame with a single column 'text' containing generated prompts.
        """
        if prompt_type == 'train':
            return pd.DataFrame(df.apply(self.generate_prompt, axis=1), columns=["text"])
        elif prompt_type == 'test':
            return pd.DataFrame(df.apply(self.generate_test_prompt, axis=1), columns=["text"])
        else:
            raise ValueError("prompt_type must be either 'train' or 'test'")



# # Assuming 'X_train_df', 'X_eval_df', and 'X_test_df' are your DataFrames containing text (and sentiment for 'X_train_df' and 'X_eval_df')
# prompt_generator = PromptGenerator()

# # Generate training and validation prompts
# X_train_df_prompts = prompt_generator.generate_dataframe_prompts(X_train_df, prompt_type='train')
# X_eval_df_prompts = prompt_generator.generate_dataframe_prompts(X_eval_df, prompt_type='train')

# # Generate test prompts
# X_test_df_prompts = prompt_generator.generate_dataframe_prompts(X_test_df, prompt_type='test')

