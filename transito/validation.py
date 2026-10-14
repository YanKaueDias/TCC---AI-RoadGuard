from transito.client import Document

EXPECTED_VEHICLES = {
    "V001": ("carro", "Carro de Ana", "ABC1D23", "Ana", "motorista", 4),
    "V002": ("moto", "Moto de Bruno", "DEF4G56", "Bruno", "piloto", 2),
    "V003": ("caminhao", "Caminhão de Carla", "GHI7J89", "Carla", "motorista", 6),
    "V004": ("moto", "Moto de Diego", "JKL0M12", "Diego", "piloto", 2),
}


class ValidationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate(results: dict[str, list[Document]]) -> None:
    try:
        validate_examples(results)
    except (KeyError, TypeError, IndexError) as error:
        raise ValidationError("Os resultados não possuem a estrutura esperada.") from error


def validate_examples(results: dict[str, list[Document]]) -> None:
    vehicles = results["01-veiculos"]
    require(len(vehicles) == 4, "A consulta geral deve retornar quatro veículos.")
    require(
        {item["codigo"]: (item["tipo"], item["modelo"]) for item in vehicles}
        == {code: values[:2] for code, values in EXPECTED_VEHICLES.items()},
        "A consulta geral deve identificar corretamente os tipos e modelos.",
    )
    documents = results["02-veiculos-com-relacoes"]
    complete = {item["codigo"]: item for item in documents}
    require(
        len(documents) == 4 and set(complete) == set(EXPECTED_VEHICLES),
        "A consulta completa deve retornar os quatro veículos sem duplicações.",
    )
    wheel_codes: set[str] = set()
    for code, values in EXPECTED_VEHICLES.items():
        vehicle_type, model, plate, person, role, wheels = values
        vehicle = complete[code]
        require(
            (vehicle["tipo"], vehicle["modelo"]) == (vehicle_type, model),
            f"Tipo ou modelo incorreto para {code}.",
        )
        require(vehicle["placas"] == [{"valor": plate}], f"Placa incorreta para {code}.")
        drivers = vehicle["condutores"]
        require(
            len(drivers) == 1 and (drivers[0]["nome"], drivers[0]["papel"]) == (person, role),
            f"Condutor ou papel incorreto para {code}.",
        )
        codes = {wheel["codigo"] for wheel in vehicle["rodas"]}
        require(
            len(vehicle["rodas"]) == wheels and len(codes) == wheels and not wheel_codes & codes,
            f"Quantidade ou vínculo de rodas incorreto para {code}.",
        )
        wheel_codes.update(codes)
    require(
        {(item["piloto"], item["moto"]) for item in results["03-pilotos"]}
        == {("Bruno", "Moto de Bruno"), ("Diego", "Moto de Diego")}
        and len(results["03-pilotos"]) == 2,
        "Os pilotos devem ser Bruno e Diego, associados às motos corretas.",
    )
    require(
        {(item["motorista"], item["veiculo"]) for item in results["04-motoristas"]}
        == {("Ana", "Carro de Ana"), ("Carla", "Caminhão de Carla")}
        and len(results["04-motoristas"]) == 2,
        "Os motoristas devem ser Ana e Carla, associados aos veículos corretos.",
    )
    require(
        results["06-pilotos-sem-capacete-cadastrado"]
        == [{"piloto": "Diego", "moto": "Moto de Diego"}],
        "A consulta sem capacete cadastrado deve retornar apenas Diego.",
    )
    require(
        {item["tipo"] for item in results["07-tipos-de-veiculo"]} == {"carro", "moto", "caminhao"}
        and len(results["07-tipos-de-veiculo"]) == 3,
        "Os subtipos de veículo devem ser carro, moto e caminhao.",
    )
    equipment = results["05-equipamentos-da-moto"]
    require(
        len(equipment) == 2
        and {(item["condutor"], item["veiculo"], item["equipamento"]) for item in equipment}
        == {
            ("Bruno", "Moto de Bruno", "Capacete de Bruno"),
            ("Bruno", "Moto de Bruno", "Bota de Bruno"),
        },
        "O encadeamento deve retornar o capacete e a bota de Bruno pela moto V002.",
    )
    expected_equipment = {
        "V001": set(),
        "V002": {("Capacete de Bruno", "capacete"), ("Bota de Bruno", "calcado")},
        "V003": set(),
        "V004": {("Tênis de Diego", "calcado")},
    }
    for code, expected in expected_equipment.items():
        equipment = complete[code]["condutores"][0]["equipamentos"]
        require(
            len(equipment) == len(expected)
            and {(item["nome"], item["tipo"]) for item in equipment} == expected,
            f"Equipamentos incorretos para o condutor de {code}.",
        )
