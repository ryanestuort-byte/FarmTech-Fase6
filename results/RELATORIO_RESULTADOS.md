# Resultados da execução — FarmTech Fase 6

## Protocolo
80 imagens: 40 bicicletas e 40 gatos; 64 treino, 8 validação e 8 teste.
YOLOv5n v7.0: duas inicializações iguais, 30 e 60 épocas, imagens 224 px,
batch 4 e semente 42. CNN do zero: 60 épocas, imagens 160 px.
Seleção dos pesos por validação. O teste é usado apenas para avaliação final.

## Detecção
| model       | split   |   precision |   recall |   map50 |   map50_95 |   inference_forward_ms | annotation_source   |
|:------------|:--------|------------:|---------:|--------:|-----------:|-----------------------:|:--------------------|
| yolo_30ep   | val     |      0.2085 |   0.35   |  0.2534 |     0.0933 |                22.7895 | Make Sense AI       |
| yolo_60ep   | val     |      0.4364 |   0.575  |  0.4952 |     0.2593 |                10.6327 | Make Sense AI       |
| yolo_padrao | val     |      0.9267 |   0.45   |  0.5811 |     0.4242 |                19.8574 | Make Sense AI       |
| yolo_30ep   | test    |      0.2014 |   0.3333 |  0.2995 |     0.0903 |                14.2445 | Make Sense AI       |
| yolo_60ep   | test    |      0.313  |   0.2667 |  0.2812 |     0.1307 |                11.1001 | Make Sense AI       |
| yolo_padrao | test    |      0.4405 |   0.4396 |  0.5198 |     0.298  |                20.8268 | Make Sense AI       |

A passagem de 30 para 60 épocas **aumentou** a mAP50–95 de validação
em +0.1660. A configuração selecionada pela validação é **yolo_60ep**.
Mais épocas não garantem desempenho melhor: compare as curvas de perda,
o pico da mAP e a época do checkpoint escolhido. Perdas de box/obj/cls
são acompanhadas no treino; não são percentuais de erro e não são
comparáveis diretamente à entropia cruzada da CNN.

No teste, a YOLO padrão obteve mAP50–95 **0.2980**,
enquanto a configuração customizada selecionada obteve
**0.1307**. Nesta amostra, customizar a rede
não assegurou melhor localização do que utilizar pesos padrão. O detector pode
acertar a classe da foto e, ainda assim, perder instâncias ou detectar caixas falsas.

## Classificação comum às abordagens
| model       |   accuracy |   error_rate |   macro_f1 |   abstentions |   latency_median_ms |   n_images |
|:------------|-----------:|-------------:|-----------:|--------------:|--------------------:|-----------:|
| yolo_30ep   |       0    |         1    |     0      |             8 |             12.2728 |          8 |
| yolo_60ep   |       0.75 |         0.25 |     0.8036 |             1 |             12.3412 |          8 |
| yolo_padrao |       0.75 |         0.25 |     0.8571 |             2 |             11.9845 |          8 |
| cnn_zero    |       1    |         0    |     1      |             0 |              5.7008 |          8 |

O detector recebe como predição da imagem a classe da caixa mais confiante
entre as duas classes-alvo, com confiança mínima 0,25. Ausência de caixa
é abstenção, contada como erro. A CNN fornece diretamente uma das duas classes.
Esta comparação avalia a identificação por imagem; somente YOLO localiza objetos.

A CNN acertou **8/8** imagens;
o detector customizado escolhido acertou **6/8**.
O treino de 30 épocas teve **8 abstenções**
com limiar 0,25. A mAP usa confiança mínima 0,001 para construir a curva PR;
portanto uma mAP acima de zero pode coexistir com várias abstenções no limiar
operacional. Mantivemos 0,25 fixo, sem ajustá-lo depois de observar o teste.

## Custo de treinamento
| model       |   epochs |   train_seconds | device   |   best_epoch |
|:------------|---------:|----------------:|:---------|-------------:|
| yolo_30ep   |       30 |         140.345 | cpu      |           29 |
| yolo_60ep   |       60 |         260.671 | cpu      |           44 |
| yolo_padrao |        0 |           0     | cpu      |          nan |
| cnn_zero    |       60 |          72.163 | cpu      |           23 |

O baseline pré-treinado não é customizado: seu custo de treinamento **neste projeto**
é zero, mas isso não representa o custo de seu pré-treinamento original.
Latência foi medida após aquecimento, com lote 1, três repetições por imagem,
no mesmo dispositivo, incluindo pré-processamento, decisão e NMS quando aplicável.
Decodificação do arquivo e carregamento dos pesos ficam fora desse intervalo.
As resoluções de entrada diferem (224/160 px), portanto a comparação representa
as configurações implantadas, não a eficiência intrínseca de cada arquitetura.
Os detectores de 30/60 épocas têm a mesma arquitetura: diferenças de latência
podem refletir o número de propostas após NMS, cache e carga concorrente do
computador. Três repetições por foto são poucas; as medições não constituem
benchmark dedicado e não justificam afirmar superioridade geral de velocidade.

## Integração e limitações
YOLO padrão exige pouco esforço quando as classes já existem no COCO; a versão
customizada acrescenta coleta, revisão de caixas e treino, mas adapta a detecção
ao escopo do cliente. A CNN tem arquitetura simples e pipeline de rótulo por imagem,
porém exige implementar treino e pré-processamento e não produz caixas.
Somente 8 imagens de teste: cada acerto altera a acurácia em 12,5 pontos percentuais.
Não há evidência suficiente para afirmar superioridade geral ou uso em produção.
O filtro de tamanho e a exclusão de imagens com ambas as classes tornam o cenário
mais simples; não avaliamos cenas mistas, negativos, vídeos ou mudança de domínio.
As imagens vêm de COCO val2017, benchmark conhecido; checkpoints pré-treinados
podem ter sido selecionados usando esse benchmark. Não tratamos esse teste como
avaliação externa totalmente independente. A CNN do zero não tem esse pré-treino.
Hashes detectam cópias exatas, mas não garantem ausência de cenas semelhantes.
Para o cliente, ampliar a base com fotos próprias, dividir por local/sessão e
avaliar negativos e cenas mistas é o próximo passo antes de implantação.

## Evidências
`comparison.png`, `confusion_matrices.png`, `predictions/` (as oito imagens por
detector), CSVs de resultados, logs, históricos e checkpoints registram a execução.
Fonte das anotações aparece explicitamente na tabela de detecção.
