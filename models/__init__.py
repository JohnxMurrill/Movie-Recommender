import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommender_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('recommender_models')