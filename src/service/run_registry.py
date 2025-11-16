# run_registry.py

from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    session_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)


class RunRegistry:
    def __init__(self, db_uri: str):
        self.engine = create_engine(db_uri, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

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
