

from service.run_registry import RunRegistry
from src.lib.logger import initialize_logging
from src.service.listener import Listener
from src.lib.config import initialize_config
from src.lib.middleware import Middleware


def main():
    config = initialize_config()
    initialize_logging(config.logging_level)
    middleware = Middleware(config)
    db = RunRegistry(config.db_uri)
    listener = Listener(middleware, config, db)
    listener.run()

if __name__ == "__main__":
    main()