import carla
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    logger.info("Connected to CARLA server")
    world = client.get_world()
    logger.info("World loaded: %s", world.get_map().name)
except Exception as e:
    logger.error("Failed to connect: %s", str(e))