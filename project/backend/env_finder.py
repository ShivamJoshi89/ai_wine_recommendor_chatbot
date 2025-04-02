from dotenv import load_dotenv
import os

# Explicitly set the path to your .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

print(os.getenv("OPENAI_API_KEY"))
print("MONGODB_URI:", os.getenv("MONGODB_URI"))