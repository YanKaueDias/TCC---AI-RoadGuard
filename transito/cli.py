import argparse
import json
import os
import sys
from pathlib import Path

from transito.application import PROJECT_ROOT, KnowledgeBase
from transito.client import Settings, TypeDBClient, TypeDBError
from transito.validation import ValidationError, validate


def serialize(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Base de conhecimento de trânsito em TypeDB.")
    parser.add_argument("acao", choices=["iniciar", "demonstrar", "validar", "consultar"])
    parser.add_argument("arquivo", nargs="?", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--database", default="transito")
    parser.add_argument("--saida", type=Path, default=PROJECT_ROOT / "resultados-verificados.json")
    args = parser.parse_args(argv)
    if args.acao == "consultar" and args.arquivo is None:
        parser.error("consultar exige o caminho de um arquivo .tql")
    if args.acao != "consultar" and args.arquivo is not None:
        parser.error("o caminho de arquivo é aceito apenas em consultar")
    try:
        query = KnowledgeBase.read(args.arquivo) if args.acao == "consultar" else None
        settings = Settings(
            url=args.url,
            database=args.database,
            username=os.environ.get("TYPEDB_USER", "admin"),
            password=os.environ.get("TYPEDB_PASSWORD", "password"),
        )
        client = TypeDBClient(settings)
        client.connect()
        base = KnowledgeBase(client)
        if args.acao == "iniciar":
            loaded = base.initialize()
            print(
                "Base criada e exemplos carregados."
                if loaded
                else "Base já contém veículos. Dados existentes preservados."
            )
        elif query is not None:
            print(serialize(client.query(query)), end="")
        else:
            results = base.examples()
            if args.acao == "validar":
                validate(results)
                args.saida.write_text(serialize(results), encoding="utf-8")
                print("Validação aprovada: herança, relações e inferências conferidas no TypeDB.")
                print(f"Resultados salvos em {args.saida}.")
            else:
                for name, answers in results.items():
                    print(f"\n{name}")
                    print(serialize(answers), end="")
    except (TypeDBError, ValidationError, OSError, ValueError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Execução interrompida.", file=sys.stderr)
        return 130
    return 0
