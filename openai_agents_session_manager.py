import uuid

from agents import Agent, SQLiteSession


class SessionManager:

    __db_path: str = "openai_agents_sessions.db"

    def __init__(self, session_id: str = str(uuid.uuid4())):
        self.session_id = session_id

    def get_session(self, agent: Agent | str | None = None) -> str:
        session_id = self.session_id
        if agent is not None:
            if isinstance(agent, Agent):
                agent = agent.name
            session_id = self.__session_id_with_postfix(agent)

        return SQLiteSession(session_id, self.__db_path)

    def __session_id_with_postfix(self, postfix: str) -> str:
        return f"{self.session_id}_{postfix}"