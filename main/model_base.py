import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QuantizedBaseModelInitializer:
    def __init__(self, model_name: str):
        """
        Initializes the QuantizedBaseModelInitializer with a model name.

        Args:
        - model_name: The name or path of the pre-trained model.
        """
        self.model_name = model_name

    def initialize_model(self) -> AutoModelForCausalLM:
        """
        Initializes and configures the model with quantization.

        Returns:
        - The initialized and quantized model.
        """
        # Get float16 data type from the torch library
        compute_dtype = getattr(torch, "float16")

        # Configuration for BitsAndBytes quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Load model weights in 4-bit format
            bnb_4bit_quant_type="nf4",  # 4-bit NormalFloat (NF4) data type
            bnb_4bit_compute_dtype=compute_dtype,  # Use float16 data type for computations
            bnb_4bit_use_double_quant=False,  # Do not use double quantization
        )

        # Load the base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=bnb_config,
        )

        # Update model configurations
        model.config.use_cache = False
        model.config.pretraining_tp = 1  # Activate accurate computation of linear layers

        return model

    def initialize_tokenizer(self) -> AutoTokenizer:
        """
        Loads the tokenizer corresponding to the model.

        Returns:
        - The initialized tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Pad the input sequence on the right side

        return tokenizer

    def initialize(self) -> tuple:
        """
        Orchestrates the initialization of the model and tokenizer.

        Returns:
        - A tuple containing the initialized model and tokenizer.
        """
        model = self.initialize_model()
        tokenizer = self.initialize_tokenizer()
        return model, tokenizer


# model_name = "NousResearch/Llama-2-7b-hf"  # Specify the model name or path

# # Create an instance of QuantizedBaseModelInitializer
# initializer = QuantizedBaseModelInitializer(model_name)

# # Initialize the model and tokenizer with quantization
# model, tokenizer = initializer.initialize()
