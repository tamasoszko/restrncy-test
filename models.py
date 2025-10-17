
import os
from agents.extensions.models.litellm_model import LitellmModel
import dotenv

dotenv.load_dotenv()

OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


openai_gpt4o = LitellmModel( model='gpt-4o', api_key=OPENAI_API_KEY, base_url=OPENAI_API_ENDPOINT)


default_model = openai_gpt4o