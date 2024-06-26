from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from tqdm.auto import tqdm

class ModelPredictor:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Initializes the ModelPredictor with a model and tokenizer.

        Args:
        - model: The pre-trained model for causal language modeling.
        - tokenizer: The tokenizer corresponding to the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pipe = pipeline(task="text-generation", 
                             model=self.model, 
                             tokenizer=self.tokenizer, 
                             max_new_tokens=1,  # Number of tokens to generate
                             temperature=0.001,   # Sampling temperature
                            )

    def predict_sentiment(self, prompt: str) -> str:
        """
        Predicts the sentiment of a given prompt.

        Args:
        - prompt: The text prompt for sentiment prediction.

        Returns:
        - The predicted sentiment as a string.
        """
        result = self.pipe(prompt)
        answer = result[0]['generated_text'].split("=")[-1].strip()

        if "positive" in answer:
            return "positive"
        elif "negative" in answer:
            return "negative"
        elif "neutral" in answer:
            return "neutral"
        else:
            # Handle cases with no tokens generated as "none"
            return "none"

    def predict(self, test: pd.DataFrame) -> list:
        """
        Generates sentiment predictions for each entry in the test DataFrame.

        Args:
        - test: DataFrame containing text prompts for sentiment prediction.

        Returns:
        - A list of predicted sentiments.
        """
        y_pred = []
        for i in tqdm(range(len(test)), desc="Predicting"):
            prompt = test.iloc[i]["text"]
            sentiment = self.predict_sentiment(prompt)
            y_pred.append(sentiment)
        return y_pred



# # Assuming 'model' and 'tokenizer' have been previously defined and initialized
# predictor = ModelPredictor(model, tokenizer)

# # Assuming 'test_df' is your test DataFrame containing text prompts in a column named "text"
# y_pred = predictor.predict(test_df)
