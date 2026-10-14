import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

TransactionType = Literal["read", "write", "schema"]
Document = dict[str, Any]


class TypeDBError(Exception):
    pass


class TypeDBUnavailable(TypeDBError):
    pass


@dataclass(frozen=True)
class Settings:
    url: str = "http://127.0.0.1:18000"
    database: str = "transito"
    username: str = "admin"
    password: str = field(default="password", repr=False)
    timeout: float = 30.0
    startup_timeout: float = 30.0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Informe uma URL HTTP ou HTTPS válida.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("A URL não deve conter credenciais, parâmetros ou fragmentos.")
        if not self.database.strip():
            raise ValueError("O nome do banco não pode ser vazio.")
        if self.timeout <= 0 or self.startup_timeout < 0:
            raise ValueError("Os tempos de conexão devem ser válidos.")


class TypeDBClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.token: str | None = None

    def connect(self) -> None:
        deadline = time.monotonic() + self.settings.startup_timeout
        while True:
            timeout = min(self.settings.timeout, max(0.1, deadline - time.monotonic()))
            try:
                response = self.request(
                    "POST",
                    "/v1/signin",
                    {"username": self.settings.username, "password": self.settings.password},
                    timeout=timeout,
                )
            except TypeDBUnavailable:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(0.5, remaining))
                continue
            if (
                not isinstance(response, dict)
                or not isinstance(response.get("token"), str)
                or not response["token"]
            ):
                raise TypeDBError("O TypeDB retornou uma autenticação inválida.")
            self.token = response["token"]
            return

    def request(
        self,
        method: str,
        path: str,
        body: Document | None = None,
        *,
        as_text: bool = False,
        timeout: float | None = None,
    ) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        request = Request(
            self.settings.url.rstrip("/") + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout or self.settings.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            try:
                details = json.loads(error.read().decode("utf-8"))
                message = (
                    details.get("message", error.reason)
                    if isinstance(details, dict)
                    else error.reason
                )
            except (ValueError, UnicodeError):
                message = error.reason
            raise TypeDBError(f"TypeDB HTTP {error.code}: {message}") from error
        except (URLError, TimeoutError, ConnectionError) as error:
            raise TypeDBUnavailable(
                "Não foi possível conectar ao TypeDB. Confira se o Docker está em execução."
            ) from error
        except UnicodeError as error:
            raise TypeDBError("O TypeDB retornou uma resposta com codificação inválida.") from error
        if as_text:
            return raw.strip()
        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except ValueError as error:
            raise TypeDBError("O TypeDB retornou uma resposta JSON inválida.") from error

    def query(self, query: str, kind: TransactionType = "read") -> list[Document]:
        result = self.request(
            "POST",
            "/v1/query",
            {
                "databaseName": self.settings.database,
                "transactionType": kind,
                "commit": kind != "read",
                "query": query,
                "queryOptions": {"includeQueryStructure": False},
            },
        )
        if not isinstance(result, dict):
            raise TypeDBError("O TypeDB retornou um resultado de consulta inválido.")
        if result.get("warning"):
            raise TypeDBError(f"Consulta incompleta: {result['warning']}")
        if result.get("answerType") == "ok":
            return []
        answers = result.get("answers")
        if not isinstance(answers, list) or not all(isinstance(answer, dict) for answer in answers):
            raise TypeDBError("O TypeDB não retornou uma lista válida de respostas.")
        return answers
