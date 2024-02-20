import logging
from logging.handlers import RotatingFileHandler

class Logger:
    def __init__(self, log_file_name, log_level=logging.INFO):
        # Assurez-vous que le logger n'est pas dupliqué
        if not logging.getLogger(__name__).hasHandlers():
            # Configurez le niveau de log global
            logging.getLogger(__name__).setLevel(log_level)

        # Créez un logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Vérifiez si des gestionnaires sont déjà configurés pour éviter les duplications
        if not self.logger.handlers:
            # Créez un gestionnaire de fichiers qui journalise même les messages de niveau INFO
            file_handler = RotatingFileHandler(log_file_name, maxBytes=10000000, backupCount=5)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Créez un gestionnaire de console pour afficher les messages
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
