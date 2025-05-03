import carla
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    client = carla.Client('localhost', 2027)
    client.set_timeout(60.0)
    world = client.load_world('Town05')
    blueprints = world.get_blueprint_library().filter('*')
    logger.info("All blueprints: %s", [bp.id for bp in blueprints])
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
    logger.info("Vehicle blueprints: %s", [bp.id for bp in vehicle_blueprints])
    tesla_blueprints = world.get_blueprint_library().filter('vehicle.tesla.model3')
    logger.info("Tesla Model 3 blueprints: %s", [bp.id for bp in tesla_blueprints])
except Exception as e:
    logger.error("Error: %s", str(e))