from cornerstone.config import Settings
from cornerstone.database import Database
from cornerstone.services.bootstrap import initialize_database, seed_demo


def main() -> None:
    settings = Settings()
    database = Database(settings.database_url)
    initialize_database(database.engine)
    with database.session_factory() as session:
        seed_demo(session, settings)


if __name__ == '__main__':
    main()
