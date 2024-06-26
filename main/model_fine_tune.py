from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from peft import PeftModel

class ModelReloader:
    def __init__(self, base_model_name: str, fine_tuned_model_id: str):
        """
        Initializes the ModelReloader with the base model name and the fine-tuned model ID.

        Args:
        - base_model_name: The name or path of the pre-trained base model.
        - fine_tuned_model_id: The ID or path of the fine-tuned model.
        """
        self.base_model_name = base_model_name
        self.fine_tuned_model_id = fine_tuned_model_id

    def collect_garbage(self):
        """
        Performs garbage collection to free up memory.
        """
        gc.collect()

    def reload_base_model(self):
        """
        Reloads the base model.

        Returns:
        - The reloaded base model.
        """
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype='auto',  # Set torch dtype to 'auto' for automatic handling
            device_map="auto",  # Automatic device mapping for optimal placement
            # quantization_config=bnb_config,  # Uncomment and adjust if quantization is needed
        )
        return base_model

    def reload_tokenizer(self):
        """
        Reloads the tokenizer corresponding to the base model.

        Returns:
        - The reloaded tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,  # Enable loading custom/remote tokenizers
        )
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token
        tokenizer.padding_side = "right"  # Set padding side
        return tokenizer

    def reload_fine_tuned_model(self, base_model):
        """
        Reloads the fine-tuned model by merging the base model with fine-tuning updates.

        Args:
        - base_model: The reloaded base model.

        Returns:
        - The reloaded fine-tuned model.
        """
        ft_model = PeftModel.from_pretrained(
            base_model,
            self.fine_tuned_model_id,
            local_files_only=True  # Load from local files
        )
        return ft_model

    def reload(self):
        """
        Orchestrates the entire process of reloading the base model, tokenizer, and fine-tuned model.

        Returns:
        - A tuple containing the reloaded fine-tuned model and tokenizer.
        """
        self.collect_garbage()
        base_model = self.reload_base_model()
        tokenizer = self.reload_tokenizer()
        ft_model = self.reload_fine_tuned_model(base_model)
        return ft_model, tokenizer


# # Specify the base model name and the fine-tuned model ID
# base_model_name = "NousResearch/Llama-2-7b-hf"
# fine_tuned_model_id = "./trained_model"

# # Create an instance of ModelReloader
# reloader = ModelReloader(base_model_name, fine_tuned_model_id)

# # Reload the fine-tuned model and tokenizer
# ft_model, tokenizer = reloader.reload()