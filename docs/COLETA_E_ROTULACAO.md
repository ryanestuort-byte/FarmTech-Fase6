# Coleta e rotulação

As 80 fotos estão em `dataset/images/train`, `val` e `test`, com 32/4/4 imagens por classe. O manifesto registra URLs, licenças e hashes. Ordem das classes: bicicleta=0; gato=1.

## Operação efetivamente realizada

Em 30/09/2026, as 80 fotos e as pré-anotações COCO foram importadas no Make Sense AI. Todas as imagens foram percorridas pela navegação do site, com verificação visual de amostras. A exportação YOLO original contém 80 arquivos TXT e 130 caixas. A auditoria confirmou classes, quantidades e coordenadas; a diferença máxima para as caixas de origem foi 0,000001, por arredondamento. Não houve desenho manual de todas as caixas.

`dataset/makesense_export.zip` preserva o arquivo exportado pelo site. `annotation_status.json` registra o hash SHA-256. Os TXT utilizados nos treinos são os dessa exportação, e `labels_coco_original` preserva os anteriores. Os treinos independentes de 30 e 60 épocas foram repetidos após a importação final.

As capturas `makesense_bicicleta.png`, `makesense_gato.png` e `makesense_export_completo.png` documentam a operação.

## Reprodução

1. Carregue as 80 imagens no Make Sense e selecione Object Detection.
2. Crie bicicleta e gato, nesta ordem.
3. Importe os TXT YOLO de `makesense_import.zip`; navegue pelas 80 imagens e confira as caixas.
4. Exporte em YOLO e use a célula de importação do notebook, que rejeita arquivos incompletos.
5. Se modificar as caixas, preserve os treinos antigos em outra pasta e execute novamente.
