from os import getenv
from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "idcardpytesseract"
    mode: str
    dbpath: str

    class Config:
        env_file = f"idcardpytesseract/envs/{getenv('MODE')}.env"
