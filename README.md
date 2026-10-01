# FarmTech Solutions — Fase 6

Projeto acadêmico de visão computacional para identificar **bicicletas e gatos**.
Comparamos YOLOv5 customizada com 30/60 épocas, YOLOv5 padrão pré-treinada e
uma CNN treinada do zero, usando 80 imagens com divisão 64/8/8.

**Integrantes:** RYAN PABLO CORREA DE PAULA (RM570587) e MATHEUS KAUÃ DA SILVA (RM569379).  
**Nome do grupo:** FarmTech-Fase6.

## Acesse a solução

- [Notebook principal — implementação, saídas e discussão](notebooks/RyanPabloCorreaDePaula_rm570587_MatheusKauaDaSilva_rm569379_pbl_fase6.ipynb).
- [Resultados e limitações da execução](results/RELATORIO_RESULTADOS.md).
- [Repositório público](https://github.com/ryanestuort-byte/FarmTech-Fase6).
- [Pacote completo com MP4 e pesos](https://drive.google.com/file/d/1eT7xmirNTZlm0X9MKgiIwQRU0ZjpX9rr/view).
- [Vídeo no Google Drive](https://drive.google.com/file/d/1CXJrr1BpVFiWmPnZxRPSy6EhmGCK7fTj/view?usp=drivesdk).
- [Vídeo não listado no YouTube](https://youtu.be/CMW7rIuIbwk).
- [Notebook no Google Drive](https://drive.google.com/file/d/1IjO5qWzAbtS81eiQyZv1wemqTSKf7OBh/view?usp=drivesdk) — disponível para leitura pelo link.
- [Dataset organizado no Google Drive](https://drive.google.com/drive/folders/1bnHR90RHoHjcWVDK6VBWQyYtN-2e3O1Y).
- [Abrir no Colab](https://colab.research.google.com/drive/1IjO5qWzAbtS81eiQyZv1wemqTSKf7OBh) — notebook com saídas reais da execução local; conexão do Drive implementada para reexecução no Colab.

O notebook é a documentação principal: nele estão coleta, rotulação, treinamento,
validação, testes, métricas e conclusões. A implementação também está em
[`src/pipeline.py`](src/pipeline.py) para facilitar inspeção do código.

## Executar

Abra o notebook no Google Colab com Python 3.11–3.12 e, preferencialmente, GPU.
Execute as células em ordem e autorize a montagem do seu Google Drive.
O notebook monta `MyDrive/FarmTech_Fase6`, prepara as imagens e salva os resultados.
As saídas registradas nesta cópia documentam a execução local; uma nova execução
produz novas medições no dispositivo utilizado.

Para execução local, crie um ambiente Python 3.12, instale `requirements.txt`
e abra o notebook a partir da pasta do projeto. O código fixa o commit oficial
do YOLOv5 v7.0 e faz uma adaptação de renderização para Pillow moderno.

## Organização

| Pasta | Conteúdo |
|---|---|
| `notebooks/` | Notebook principal para a correção |
| `src/` | Código Python comentado |
| `dataset/` | Imagens, caixas YOLO, manifesto e referências de licença |
| `results/` | Históricos, métricas, gráficos, pesos e prints de teste |
| `docs/` | Evidências de rotulação e status da entrega |
| `video/` | Demonstração narrada, com até cinco minutos |

As imagens vêm do COCO 2017. Atribuição e licença de cada foto constam no
manifesto e em `dataset/coco_licenses.json`. YOLOv5 v7.0 é software da Ultralytics,
distribuído sob GPL-3.0. Código e documentação foram preparados com assistência
de IA e devem ser compreendidos e revisados pelos integrantes.

## Finalização para a FIAP

Enviar o link do GitHub pelo portal da FIAP. Conferir o status da rotulação em `docs/STATUS_ENTREGA.md`.
Salvar todas as saídas do notebook e não realizar commits após o prazo de entrega.

As opções “Ir Além” são extras e não estão incluídas nas Entregas 1 e 2 deste projeto.

## Rotulação concluída

As pré-anotações COCO foram importadas no Make Sense. As 80 imagens foram percorridas no site, com conferência visual de amostras, e a exportação real em YOLO contém **80 TXT e 130 caixas**. O ZIP original, o hash e o status estão em `dataset/makesense_export.zip` e `annotation_status.json`. Não houve desenho manual de todas as caixas: os rótulos partem das anotações da fonte. Os treinos de 30/60 épocas desta versão foram repetidos com os TXT exportados pelo Make Sense.
