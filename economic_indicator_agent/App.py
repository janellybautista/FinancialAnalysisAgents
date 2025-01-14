import os
import openai
import time
import logging
import json

# If you need these from your original code
from economic_indicator_agent.data_fetcher import DataFetcher, get_api_key

DEFAULT_MODEL_ID = "gpt-4o-mini"
MAX_SAMPLES = 3


class OpenAIClient:
    _instance = None  # Holds the single instance (None until created)

    # def __new__(cls, api_key=None):
    #     """
    #     Enforces a single instance. If _instance is None, we create a new one;
    #     otherwise, we return the existing instance.
    #     """
    #     # If no instance exists yet...
    #     if cls._instance is None:
    #         if not api_key:
    #             raise ValueError("API key is required the first time MyOpenAIClient is instantiated.")
    #         # Create a new instance via the superclass __new__
    #         cls._instance = super(OpenAIClient, cls).__new__(cls)
    #     return cls._instance

    def __init__(self, api_key=None):
        """
        __init__ will be called every time MyOpenAIClient() is invoked.
        We use a guard (_initialized) to avoid re-running logic if
        we've already done so once.
        """
        if not hasattr(self, "_initialized"):
            if not api_key:
                # If somehow we get here without an API key the first time,
                # raise an error (should not happen if we check in __new__).
                raise ValueError("API key is required.")
            self.api_key = api_key
            if self._instance is None:
                # openai.api_key = api_key
                self._instance = openai.OpenAI(api_key=api_key)
            self._initialized = True  # Mark the instance as fully initialized

    def load_fine_tune_id(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data.get('fine_tune_id')
        return None

    def save_fine_tune_id(self, file_path, fine_tune_id):
        with open(file_path, 'w') as file:
            json.dump({'fine_tune_id': fine_tune_id}, file)

    def fine_tune_model(self, training_file_path, model='gpt-4o-mini-2024-07-18', id_file='fine_tune_id.json'):
        """
        Upload a file and create (or resume) a fine-tune job.
        Monitor it until completion.
        Return the fine-tuned model name on success, or None on failure.
        """
        # Ensure the module-level API key is set
        openai.api_key = self.api_key
        print(type(self._instance))
        # 1. Check if we already have a stored fine-tune job ID
        fine_tune_id = self.load_fine_tune_id(id_file)
        if fine_tune_id:

            try:
                status_response = self._instance.fine_tuning.jobs.retrieve(fine_tuning_job_id=fine_tune_id)
                status = status_response.status
                if status == 'succeeded':
                    fine_tuned_model = status_response.fine_tuned_model
                    print(f"Fine-tuning already completed. Model: {fine_tuned_model}")
                    return fine_tuned_model
                elif status in ['pending', 'running']:
                    print("Fine-tuning is still in progress.")
                    return None
                else:
                    print(f"Fine-tuning job status: {status}. Proceeding with a new fine-tuning job.")
            except openai.BadRequestError:
                print("Invalid fine-tune ID. Starting a new fine-tuning job.")
            except Exception as e:
                print(f"An error occurred retrieving status: {e}")
                return None

        # 2. Upload the training file
        print("Uploading training file...")
        try:
            with open(training_file_path, 'rb') as f:
                training_file = self._instance.files.create(file=f, purpose='fine-tune')
            #     training_file = self._instance.fine_tuning.jobs.create(
            #         training_file=f, model=model
            #     )
            training_file_id = training_file.id
            print(f"Training file uploaded with ID: {training_file_id}")
        except Exception as e:
            print(f"Failed to upload training file: {e}")
            return None

        # 3. Initiate fine-tuning
        print("Starting fine-tuning job...")
        try:
            fine_tune_response = self._instance.fine_tuning.jobs.create(
                model=model,
                training_file=training_file_id
            )
            fine_tune_id = fine_tune_response.id
            print(f"Fine-tuning job started with ID: {fine_tune_id}")
            self.save_fine_tune_id(id_file, fine_tune_id)
        except Exception as e:
            print(f"Failed to start fine-tuning job: {e}")
            return None
        except json.JSONDecodeError:
            print("Failed to decode JSON. The API response might be empty or invalid.")
            return None

        # 4. Monitor the job
        print("Monitoring fine-tuning progress...")
        while True:
            try:
                status_response = self._instance.fine_tuning.jobs.retrieve(fine_tuning_job_id=fine_tune_id)
                status = status_response.status
                print(f"Current status: {status}")

                if status == 'succeeded':
                    fine_tuned_model = status_response.fine_tuned_model
                    print(f"Fine-tuning succeeded. Model: {fine_tuned_model}")
                    return fine_tuned_model
                elif status == 'failed':
                    print("Fine-tuning failed.")
                    return None

                # Wait before checking again
                time.sleep(60)
            except Exception as e:
                print(f"An error occurred while monitoring: {e}")
                logging.error(f"Error during fine-tune monitoring: {e}")
                return None

    def completions_with_backoff(self, model, messages, max_retries=3, initial_delay=1, max_delay=10):
        """
        Calls the OpenAI completion API with exponential backoff for retries.

        Parameters:
        - model (str): The model to use for completion (e.g., "text-davinci-003" or a fine-tuned model).
        - messages (list[dict]): The conversation history or input prompts for the completion.
        - max_retries (int): Maximum number of retries in case of failure.
        - initial_delay (float): Initial delay between retries in seconds.
        - max_delay (float): Maximum delay between retries in seconds.

        Returns:
        - dict: The response from the OpenAI API.

        Raises:
        - Exception: If the maximum number of retries is exceeded.
        """
        retry_count = 0
        delay = initial_delay

        while retry_count < max_retries:
            try:
                # Call OpenAI's Chat Completion API
                response = self._instance.chat.completions.create(model=model,
                messages=messages)
                return response  # Return the successful response

            except openai.RateLimitError as e:
                # Handle rate limit errors
                retry_count += 1
                print(f"RateLimitError: {e}. Retrying in {delay} seconds...")
            except openai.APIError as e:
                # Handle server errors
                retry_count += 1
                print(f"APIError: {e}. Retrying in {delay} seconds...")
            except Exception as e:
                # Handle other exceptions
                print(f"Unexpected error: {e}. Retrying in {delay} seconds...")
                retry_count += 1

            # Wait before the next retry with exponential backoff
            time.sleep(delay)
            delay = min(delay * 2, max_delay)  # Increase delay with a cap at max_delay

        # If all retries fail, raise an exception
        raise Exception(f"Max retries exceeded. Unable to complete request to model {model}.")

    def get_existing_fine_tuned_model(self):
        """
        Retrieves the ID of an existing fine-tuned model with a 'succeeded' status.

        Returns:
            str: The ID of the fine-tuned model if found, otherwise None.
        """
        try:
            # List all models available to your account
            models = self._instance.models.list()

            # Iterate through the models to find fine-tuned ones with 'succeeded' status
            for model in models['data']:
                # Check if the model is a fine-tuned model and its status is 'succeeded'
                if model.get('fine_tuned') and model.get('status') == 'succeeded':
                    return model['id']

            # If no suitable model is found, return None
            return None

        except Exception as e:
            print(f"An error occurred while retrieving fine-tuned models: {e}")
            return None

if __name__ == "__main__":

    # 1. Get your API key
    openAIKey = get_api_key("openAI")
    if not openAIKey:
        raise ValueError("OPENAI_API_KEY is not set.")

    # 2. Instantiate the client
    client = OpenAIClient(api_key=openAIKey)

    # 3. Fine-tune if needed
    training_file_path = "fine-tuning.jsonl"

    fine_tuned_model = client.get_existing_fine_tuned_model()

    if not fine_tuned_model:
        fine_tuned_model = client.fine_tune_model(
            training_file_path=training_file_path
        )

    # 4. If we got a fine-tuned model back, use it
    if fine_tuned_model:
        messages = [
            {"role": "system", "content": "You are an expert financial analyst."},
            {"role": "user", "content": "Incoming CPI PREDICTION AND FINANCIAL ADVICE?"}
        ]

        response = client.completions_with_backoff(
            model=fine_tuned_model,
            messages=messages
        )

        if response and response.choices:
            print(response.choices[0].message.content)
        else:
            print("No response from the fine-tuned model.")
