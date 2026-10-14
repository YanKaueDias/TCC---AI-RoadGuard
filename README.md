# Protótipo de visão computacional e base de conhecimento de trânsito

O projeto reúne a atividade 2 (base de conhecimento em TypeDB) e um protótipo local da atividade 1. O protótipo visual funciona sem iniciar o TypeDB e grava os resultados em arquivos JSON; ele não altera o schema nem os dados existentes.

## Atividade 1: análise de imagens e vídeos

Requisitos: Python 3.10 ou superior. No PowerShell, na pasta do projeto:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-visao.txt
```

Exemplos:

```powershell
python visao.py imagem "C:\caminho\para\imagem.jpg"
python visao.py video "C:\caminho\para\video.mp4"
```

Por padrão, a imagem anotada será salva ao lado da entrada com o sufixo `_anotada`; o vídeo será salvo como MP4 com o sufixo `_anotado`. Para cada arquivo, também será criado um relatório JSON com sufixo `_relatorio`.

É possível indicar caminhos de saída, pesos YOLO compatíveis com as classes COCO e ajustar confiança ou resolução:

```powershell
python visao.py imagem "C:\caminho\para\imagem.jpg" --saida "C:\caminho\para\resultado.jpg" --confianca 0.3 --tamanho 640
python visao.py video "C:\caminho\para\video.mp4" --saida "C:\caminho\para\resultado.mp4"
```

O modelo `yolo11n.pt` é baixado automaticamente pelo Ultralytics na primeira execução. A instalação e o primeiro uso precisam de acesso à internet. CPU é suficiente para experimentar, mas vídeos podem ser processados lentamente; uma GPU compatível pode acelerar a análise. O modelo pré-treinado e o rastreamento entre quadros seguem os recursos descritos na [documentação do YOLO11](https://docs.ultralytics.com/models/yolo11/) e na [documentação de rastreamento](https://docs.ultralytics.com/modes/track/).

### O que o protótipo detecta e descreve

- Objetos das classes COCO selecionadas: pessoa, bicicleta, carro, moto, ônibus, caminhão, semáforo e placa de pare.
- Em imagens, salva caixas delimitadoras, confiança e posição no quadro. Também estima linhas retas da pista e, quando há ao menos três limites visíveis, estima se um veículo está na faixa esquerda, central ou direita.
- Em vídeos, usa ByteTrack para associar detecções entre quadros, registra IDs e resume quantos quadros cada objeto apareceu.
- Pode registrar um evento de **possível passagem visual por sinal** quando a trajetória de um veículo rastreado cruza a altura da caixa de um semáforo ou placa de pare detectado.
- As linhas de pista são estimadas com Canny e Hough, sem um modelo treinado especificamente para faixas.

### Limitações

A estimativa de faixa procura linhas retas na região inferior da imagem. Pode confundir marcações com sombras, bordas ou outros contrastes e não acompanha faixas curvas; depende de uma câmera frontal e de marcações visíveis. Quando não encontra limites suficientes, informa apenas a posição no quadro. O protótipo não segmenta a pista, não lê a cor ou o estado do semáforo e não decide se houve infração. A passagem por sinal é uma regra geométrica aproximada para vídeo frontal em que o veículo se desloca para a parte inferior do quadro; depende da câmera, da perspectiva, de o objeto estar visível e de o rastreador manter o mesmo ID. Ela não confirma que o veículo passou fisicamente pelo sinal. O modelo pode falhar com objetos distantes, parcialmente encobertos, iluminação ruim ou cenas fora dos exemplos comuns do conjunto COCO. As descrições são regras sobre detecções, não uma compreensão semântica completa do vídeo.

## Atividade 2: base de conhecimento TypeDB

A CLI existente continua disponível pelo arquivo `prototipo.py`. Com Docker em execução:

```powershell
docker compose up -d
python prototipo.py iniciar
python prototipo.py demonstrar
python prototipo.py validar
```

Para executar uma consulta específica:

```powershell
python prototipo.py consultar consultas\01-veiculos.tql
```

As análises visuais não precisam de Docker ou TypeDB. Os fatos existentes de veículos, pessoas, equipamentos e relações permanecem separados das observações de imagens e vídeos.
