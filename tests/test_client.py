import io
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from transito.client import Settings, TypeDBClient, TypeDBError, TypeDBUnavailable


class ClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TypeDBClient(Settings(startup_timeout=0))

    @staticmethod
    def response(value: object) -> MagicMock:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(value).encode()
        return response

    def test_rejects_empty_authentication_token(self) -> None:
        with patch("transito.client.urlopen", return_value=self.response({"token": ""})):
            with self.assertRaisesRegex(TypeDBError, "autenticação inválida"):
                self.client.connect()

    def test_rejects_truncated_results(self) -> None:
        result = {"answerType": "conceptDocuments", "answers": [], "warning": "limit reached"}
        with patch("transito.client.urlopen", return_value=self.response(result)):
            with self.assertRaisesRegex(TypeDBError, "Consulta incompleta"):
                self.client.query("match $v isa veiculo;")

    def test_does_not_retry_a_write_after_connection_failure(self) -> None:
        with patch("transito.client.urlopen", side_effect=URLError("connection lost")) as request:
            with self.assertRaises(TypeDBUnavailable):
                self.client.query("insert $p isa pessoa;", "write")
        self.assertEqual(request.call_count, 1)

    def test_commits_writes_and_keeps_reads_read_only(self) -> None:
        for kind, commit in (("read", False), ("write", True), ("schema", True)):
            with self.subTest(kind=kind):
                with patch(
                    "transito.client.urlopen", return_value=self.response({"answerType": "ok"})
                ) as request:
                    self.client.query("query", kind)
                payload = json.loads(request.call_args.args[0].data)
                self.assertEqual(payload["transactionType"], kind)
                self.assertIs(payload["commit"], commit)

    def test_reports_http_error_with_non_json_response(self) -> None:
        error = HTTPError("http://localhost", 503, "Unavailable", {}, io.BytesIO(b"unavailable"))
        with patch("transito.client.urlopen", side_effect=error):
            with self.assertRaisesRegex(TypeDBError, "HTTP 503"):
                self.client.request("GET", "/v1/databases")

    def test_rejects_invalid_json(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"invalid"
        with patch("transito.client.urlopen", return_value=response):
            with self.assertRaisesRegex(TypeDBError, "JSON inválida"):
                self.client.request("GET", "/v1/databases")

    def test_password_is_not_exposed_in_settings_representation(self) -> None:
        self.assertNotIn("example-credential", repr(Settings(password="example-credential")))
