# run_registry.py

from sqlite3 import OperationalError
from time import time
import logging
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    session_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)


class RunRegistry:
    def __init__(self, db_uri: str):
        self.engine = create_engine(
            db_uri, echo=False, pool_pre_ping=True, pool_size=5, max_overflow=10
        )
        self.Session = sessionmaker(bind=self.engine)
        self._initialize_db_with_retries()

    def _initialize_db_with_retries(self):
        """Initialize the database, retrying if the DB is not ready."""
        max_retries = 10
        wait_seconds = 3

        for attempt in range(max_retries):
            try:
                logging.info(
                    f"Trying to connect to the DB, attempt {attempt + 1} of {max_retries}..."
                )
                Base.metadata.create_all(self.engine)
                logging.info("Sucesfully connected to the DB.")
                return
            except OperationalError as e:
                logging.warning(
                    f"DB is not ready yet: {e}, waiting {wait_seconds} seconds before retrying..."
                )
                time.sleep(wait_seconds)
            except Exception as e:
                logging.error(f"Unexpected error during DB initialization: {e}")
                raise e

        raise Exception("Failed to connect to the DB after multiple attempts.")

    def save_run_id(self, session_id: str, run_id: str):
        with self.Session() as session:
            existing = session.get(Run, session_id)
            if existing:
                existing.run_id = run_id
            else:
                session.add(Run(session_id=session_id, run_id=run_id))
            session.commit()

    def get_run_id(self, session_id: str):
        with self.Session() as session:
            row = session.get(Run, session_id)
            return row.run_id if row else None
