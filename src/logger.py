import logging
from logging.handlers import RotatingFileHandler
import os

class Logger:
    def __init__(self, log_file_name, log_level=logging.INFO):
        # Créez un logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Créez un gestionnaire de fichiers qui journalise même les messages de niveau INFO
        handler = RotatingFileHandler(log_file_name, maxBytes=10000000, backupCount=5)
        handler.setLevel(log_level)

        # Créez un formateur de journalisation et ajoutez-le au gestionnaire
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Ajoutez le gestionnaire au logger
        self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger
