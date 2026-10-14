import os
import unittest
from dataclasses import replace
from uuid import uuid4

from transito.application import KnowledgeBase
from transito.client import Settings, TypeDBClient, TypeDBError
from transito.validation import validate


@unittest.skipUnless(
    os.environ.get("TYPEDB_TEST_URL"), "Defina TYPEDB_TEST_URL para testar o banco."
)
class IntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            url=os.environ["TYPEDB_TEST_URL"],
            database=f"teste-transito-{uuid4().hex}",
            username=os.environ.get("TYPEDB_USER", "admin"),
            password=os.environ.get("TYPEDB_PASSWORD", "password"),
        )
        self.client = TypeDBClient(self.settings)
        self.client.connect()
        path = f"/v1/databases/{self.settings.database}"
        self.client.request("POST", path)
        self.addCleanup(self.client.request, "DELETE", path)
        self.base = KnowledgeBase(self.client)
        self.assertTrue(self.base.initialize())

    def query(self, name: str) -> list[dict]:
        path = self.base.root / "consultas" / name
        return self.client.query(self.base.read(path))

    def test_activity_requirements(self) -> None:
        validate(self.base.examples())

    def test_initialization_preserves_existing_examples(self) -> None:
        before = self.base.examples()
        self.assertFalse(self.base.initialize())
        after = self.base.examples()
        validate(after)
        for name in before:
            with self.subTest(query=name):
                self.assertCountEqual(before[name], after[name])

    def test_inference_changes_when_facts_change(self) -> None:
        self.client.query(
            'match $v isa moto, has codigo "V002"; '
            'insert $p isa pessoa, has codigo "P005", has nome "Eva"; '
            "conducao (condutor: $p, veiculo-conduzido: $v);",
            "write",
        )
        self.assertIn({"piloto": "Eva", "moto": "Moto de Bruno"}, self.query("03-pilotos.tql"))
        self.assertIn(
            {"piloto": "Eva", "moto": "Moto de Bruno"},
            self.query("06-pilotos-sem-capacete-cadastrado.tql"),
        )
        self.client.query(
            'match $p isa pessoa, has codigo "P005"; '
            '$e isa capacete, has codigo "E001"; '
            "insert uso-equipamento (usuario: $p, item: $e);",
            "write",
        )
        self.assertEqual(
            [{"piloto": "Diego", "moto": "Moto de Diego"}],
            self.query("06-pilotos-sem-capacete-cadastrado.tql"),
        )

    def test_incompatible_schema_is_preserved(self) -> None:
        settings = replace(self.settings, database=f"teste-transito-{uuid4().hex}")
        client = TypeDBClient(settings)
        client.connect()
        path = f"/v1/databases/{settings.database}"
        client.request("POST", path)
        self.addCleanup(client.request, "DELETE", path)
        client.query("define entity externo;", "schema")
        before = client.request("GET", f"{path}/schema", as_text=True)
        with self.assertRaisesRegex(TypeDBError, "outro esquema"):
            KnowledgeBase(client).initialize()
        self.assertEqual(before, client.request("GET", f"{path}/schema", as_text=True))
