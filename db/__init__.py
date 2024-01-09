import sqlite3

def create_database():
    # Connexion à la base de données SQLite
    connection = sqlite3.connect('./reports/db_evaluation/evaluation.db')
    cursor = connection.cursor()

    # Création de la table model_evaluation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_evaluation (
            id INTEGER PRIMARY KEY,
            mae REAL,
            rmse REAL,
            r2 REAL,
            date TEXT
        );
    """)

    # Sauvegarde des changements et fermeture de la connexion
    connection.commit()
    connection.close()

if __name__ == "__main__":
    create_database()
