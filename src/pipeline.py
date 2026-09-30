"""FarmTech Fase 6: coleta rastreável, validação, YOLOv5 e CNN.

As funções são também incorporadas ao notebook, que funciona sem este arquivo.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests
import yaml
from PIL import Image, ImageDraw

SEED = 42
NAMES = ['bicicleta', 'gato']
COCO_IDS = {2: 0, 17: 1}  # IDs oficiais COCO -> classes locais.
COCO_MODEL_IDS = {1: 0, 15: 1}  # Índices contíguos usados pelo YOLO COCO.
REPO_COMMIT = '915bbf294bb74c859f0b41f1c23bc395014ea679'


def download(url, target):
    """Baixa com timeout/repetição e substitui arquivo só após conclusão."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        return target
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with target.with_suffix(target.suffix + '.part').open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            target.with_suffix(target.suffix + '.part').replace(target)
            return target
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)


def normalized_box(box, width, height):
    """Converte COCO xywh para YOLO cxcywh; recorta coordenadas na imagem."""
    x, y, w, h = box
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Caixa sem área após recorte.')
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def prepare_dataset(root, cache):
    """Seleciona 40 fotos por classe do COCO val2017, sem classes-alvo mistas.

    val2017 é a origem; train/val/test abaixo são as novas divisões acadêmicas.
    As caixas COCO são pré-anotações, não substituem a etapa Make Sense.
    """
    root, cache = Path(root), Path(cache)
    if (root / 'manifest.csv').exists():
        return audit_dataset(root)
    archive = download('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',
                       cache / 'annotations_trainval2017.zip')
    with zipfile.ZipFile(archive) as z:
        coco = json.loads(z.read('annotations/instances_val2017.json'))
    by_image = defaultdict(list)
    for ann in coco['annotations']:
        if ann['category_id'] in COCO_IDS:
            by_image[ann['image_id']].append(ann)
    image_info = {x['id']: x for x in coco['images']}
    rows, seen_hashes = [], set()
    rng = random.Random(SEED)
    for category, class_id in COCO_IDS.items():
        candidates = []
        for image_id, anns in sorted(by_image.items()):
            if {a['category_id'] for a in anns} != {category}:
                continue  # Classificação exige um único rótulo por imagem.
            if any(a.get('iscrowd', 0) for a in anns):
                continue  # Exclui grupos ambíguos.
            info = image_info[image_id]
            if max(a['area'] for a in anns) / (info['width'] * info['height']) < .02:
                continue  # Evita só objetos minúsculos; filtro documentado.
            candidates.append(image_id)
        rng.shuffle(candidates)
        collected = 0
        for image_id in candidates:
            if collected == 40:
                break
            info = image_info[image_id]
            split = 'train' if collected < 32 else 'val' if collected < 36 else 'test'
            filename = f'{image_id:012d}.jpg'
            url = 'https://s3.amazonaws.com/images.cocodataset.org/val2017/' + filename
            image = download(url, root / 'images' / split / filename)
            with Image.open(image) as im:
                im.verify()  # Confirma integridade antes de aceitar a imagem.
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen_hashes:
                image.unlink()
                continue
            seen_hashes.add(digest)
            lines = []
            for ann in by_image[image_id]:
                box = normalized_box(ann['bbox'], info['width'], info['height'])
                lines.append(str(class_id) + ' ' + ' '.join(f'{v:.8f}' for v in box))
            label = root / 'labels' / split / (image.stem + '.txt')
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text('\n'.join(lines) + '\n')
            rows.append(dict(image_id=image_id, filename=filename, split=split,
                             class_id=class_id, classe=NAMES[class_id], sha256=digest,
                             source_url=url, flickr_url=info.get('flickr_url', ''),
                             license_id=info.get('license'), width=info['width'], height=info['height']))
            collected += 1
            print(f'{NAMES[class_id]}: {collected}/40 ({split})', flush=True)
        if collected != 40:
            raise RuntimeError('Não há 40 imagens elegíveis/baixáveis para a classe.')
    pd.DataFrame(rows).to_csv(root / 'manifest.csv', index=False)
    (root / 'coco_licenses.json').write_text(json.dumps(coco['licenses'], indent=2))
    write_data_yaml(root)
    export_annotation_package(root)
    return audit_dataset(root)


def write_data_yaml(root):
    root = Path(root).resolve()
    config = dict(path=str(root), train='images/train', val='images/val', test='images/test',
                  nc=2, names=NAMES)
    (root / 'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True))


def export_annotation_package(root):
    """Cria arquivo com fotos e pré-anotações para importar no Make Sense."""
    root = Path(root)
    with zipfile.ZipFile(root / 'makesense_import.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for split in ['train', 'val', 'test']:
            for image in sorted((root / 'images' / split).glob('*.jpg')):
                z.write(image, 'images/' + image.name)
                label = root / 'labels' / split / (image.stem + '.txt')
                z.write(label, 'labels/' + label.name)
        z.writestr('labels/classes.txt', '\n'.join(NAMES))
        z.writestr('labels/labels.txt', '\n'.join(NAMES))  # Definição exigida pelo Make Sense.


def import_makesense(root, zip_path):
    """Importa YOLO exportado pelo Make Sense, validando antes de substituir."""
    root = Path(root)
    manifest = pd.read_csv(root / 'manifest.csv')
    required = {Path(f).stem for f in manifest.filename}
    with zipfile.ZipFile(zip_path) as z:
        labels = {Path(n).stem: z.read(n).decode('utf-8-sig') for n in z.namelist()
                  if n.endswith('.txt') and Path(n).stem in required}
    missing = required - labels.keys()
    if missing:
        raise ValueError(f'Exportação incompleta: faltam {len(missing)} rótulos. Revise as 80 imagens.')
    for row in manifest.itertuples():
        validate_label(labels[Path(row.filename).stem], row.class_id)
    backup = root / 'labels_coco_original'
    if not backup.exists():
        shutil.copytree(root / 'labels', backup)
    for row in manifest.itertuples():
        (root / 'labels' / row.split / (Path(row.filename).stem + '.txt')).write_text(
            labels[Path(row.filename).stem])
    for cached in root.rglob('*.cache'):
        cached.unlink()
    (root / 'makesense_export.zip').write_bytes(Path(zip_path).read_bytes())
    (root / 'annotation_status.json').write_text(json.dumps({
        'source': 'Make Sense AI export', 'export_sha256': hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(),
        'images': 80, 'class_order': NAMES}, indent=2))
    return audit_dataset(root)


def validate_label(text, expected_class):
    lines = [l.split() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError('Imagem positiva sem caixa anotada.')
    for line in lines:
        if len(line) != 5:
            raise ValueError('YOLO exige cinco campos, sem coluna de confiança.')
        values = np.asarray([float(x) for x in line])
        c, x, y, w, h = values
        if not np.isfinite(values).all() or c != int(expected_class):
            raise ValueError('Classe incorreta ou valores não finitos.')
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError('Coordenadas YOLO fora dos limites.')
        if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
            raise ValueError('Caixa ultrapassa a imagem.')


def audit_dataset(root):
    root = Path(root)
    df = pd.read_csv(root / 'manifest.csv')
    assert len(df) == 80 and df.image_id.nunique() == 80 and df.sha256.nunique() == 80
    expected = {'train': 32, 'val': 4, 'test': 4}
    for class_id in [0, 1]:
        for split, count in expected.items():
            assert len(df[(df.class_id == class_id) & (df.split == split)]) == count
    for row in df.itertuples():
        image = root / 'images' / row.split / row.filename
        assert hashlib.sha256(image.read_bytes()).hexdigest() == row.sha256
        validate_label((root / 'labels' / row.split / (image.stem + '.txt')).read_text(), row.class_id)
    write_data_yaml(root)  # Corrige caminhos absolutos após cópia para o Drive.
    print('Auditoria: 80 imagens íntegras, divisão 64/8/8 e nenhuma duplicata exata.')
    return df.groupby(['classe', 'split']).size().unstack().reindex(columns=['train', 'val', 'test'])


def show_samples(root, output):
    """Gera evidência com caixas de referência, sem chamar isso de predição."""
    root, output = Path(root), Path(output)
    df = pd.read_csv(root / 'manifest.csv')
    rows = df.groupby(['class_id', 'split'], sort=True).head(1)
    canvas = Image.new('RGB', (1080, 720), '#101827')
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows.itertuples()):
        im = Image.open(root / 'images' / row.split / row.filename).convert('RGB')
        d = ImageDraw.Draw(im)
        for line in (root / 'labels' / row.split / (Path(row.filename).stem + '.txt')).read_text().splitlines():
            c, x, y, w, h = map(float, line.split())
            box = ((x-w/2)*im.width, (y-h/2)*im.height, (x+w/2)*im.width, (y+h/2)*im.height)
            d.rectangle(box, outline='#2fe6a8', width=4)
        im.thumbnail((344, 294))
        px, py = (i % 3) * 360 + 8, (i // 3) * 360 + 44
        canvas.paste(im, (px, py))
        draw.text((px, py - 28), f'{row.classe} | {row.split} | {row.image_id}', fill='white')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def setup_yolo(repo):
    repo = Path(repo).resolve()
    if not repo.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v7.0',
                        'https://github.com/ultralytics/yolov5.git', str(repo)], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == REPO_COMMIT, 'Revisão YOLOv5 diferente da fixada.'
    download('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt', repo / 'yolov5n.pt')
    # Compatibilidade de renderização: Pillow >=10 removeu getsize().
    plots = repo / 'utils/plots.py'
    source = plots.read_text()
    for value in ['label', 'text']:
        source = source.replace(f'self.font.getsize({value})',
            f'(self.font.getbbox({value})[2], self.font.getbbox({value})[3])')
    plots.write_text(source)
    return repo


def train_yolo(root, repo, results, epochs, device='cpu', imgsz=224):
    """Duas execuções independentes partem do MESMO checkpoint pré-treinado."""
    root, repo, results = map(lambda p: Path(p).resolve(), [root, repo, results])
    name = f'yolo_{epochs}ep'
    run = results / 'train' / name
    if (run / 'execution.json').exists():
        meta = json.loads((run / 'execution.json').read_text())
        meta['weights'] = str(run / 'weights/best.pt')
        previous = pd.read_csv(run / 'results.csv')
        previous.columns = previous.columns.str.strip()
        meta['best_epoch'] = int((.1*previous['metrics/mAP_0.5']+.9*previous['metrics/mAP_0.5:0.95']).idxmax())+1
        return meta
    if run.exists():
        raise RuntimeError(f'Execução parcial em {run}; renomeie para preservá-la antes de repetir.')
    results.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(repo / 'train.py'), '--data', str(root / 'data.yaml'),
           '--weights', str(repo / 'yolov5n.pt'), '--epochs', str(epochs), '--imgsz', str(imgsz),
           '--batch-size', '4', '--workers', '0', '--device', device, '--seed', str(SEED),
           '--patience', str(epochs+1), '--project', str(results / 'train'), '--name', name,
           '--exist-ok']
    started = time.perf_counter()
    with (results / f'{name}.log').open('w') as log:
        process = subprocess.Popen(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(10)
            csv = run / 'results.csv'
            completed = len(pd.read_csv(csv)) if csv.exists() else 0
            print(f'{name}: {completed}/{epochs} épocas registradas', flush=True)
        if process.returncode:
            raise RuntimeError(f'Treino falhou. Consulte {log.name}.')
    history = pd.read_csv(run / 'results.csv')
    history.columns = history.columns.str.strip()
    assert len(history) == epochs, 'Treino não completou a quantidade solicitada de épocas.'
    meta = dict(model=name, epochs=epochs, train_seconds=time.perf_counter()-started,
                weights=str(run / 'weights/best.pt'), device=device, imgsz=imgsz,
                seed=SEED, repo_commit=REPO_COMMIT,
                best_epoch=int((.1*history['metrics/mAP_0.5']+.9*history['metrics/mAP_0.5:0.95']).idxmax())+1)
    (run / 'execution.json').write_text(json.dumps(meta, indent=2))
    return meta


def initialize_yolo_imports(repo):
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch
    torch.set_num_threads(4)


def evaluate_detection(root, repo, results, weights, name, split, standard=False, device='cpu'):
    """mAP com caixas reais. Baseline usa rótulos remapeados para 80 classes."""
    initialize_yolo_imports(repo)
    from val import run
    root, results = Path(root).resolve(), Path(results).resolve()
    data = root / 'data.yaml'
    if standard:
        target = root / 'baseline_coco'
        for s in ['train', 'val', 'test']:
            (target / 'images' / s).mkdir(parents=True, exist_ok=True)
            (target / 'labels' / s).mkdir(parents=True, exist_ok=True)
            for image in (root / 'images' / s).glob('*.jpg'):
                if not (target / 'images' / s / image.name).exists():
                    shutil.copy2(image, target / 'images' / s / image.name)
                label = root / 'labels' / s / (image.stem+'.txt')
                lines = []
                for line in label.read_text().splitlines():
                    parts = line.split()
                    parts[0] = str({0: 1, 1: 15}[int(parts[0])])
                    lines.append(' '.join(parts))
                (target / 'labels' / s / label.name).write_text('\n'.join(lines)+'\n')
        for p in target.rglob('*.cache'):
            p.unlink()
        original = yaml.safe_load((Path(repo) / 'data/coco.yaml').read_text())
        config = dict(path=str(target), train='images/train', val='images/val', test='images/test',
                      nc=80, names=original['names'])
        data = target / 'data.yaml'
        data.write_text(yaml.safe_dump(config))
    metrics, maps, speed = run(data=str(data), weights=str(weights), imgsz=224, batch_size=4,
                              task=split, device=device, workers=0, half=False,
                              project=str(results / 'evaluation'), name=f'{name}_{split}',
                              exist_ok=True, plots=True)
    # val.py retorna P, R, mAP50, mAP50-95 e três losses (zeros fora do treino).
    p, r, map50, map95 = map(float, metrics[:4])
    record = dict(model=name, split=split, precision=p, recall=r, map50=map50, map50_95=map95,
                  inference_forward_ms=float(speed[1]),
                  annotation_source='Make Sense AI' if (root/'annotation_status.json').exists() else 'COCO (pré-anotação)')
    out = results / 'evaluation' / f'{name}_{split}' / 'metrics.json'
    out.write_text(json.dumps(record, indent=2))
    return record


def yolo_image_predictions(root, repo, weights, standard=False, device='cpu', split='test'):
    """Classificação derivada: classe da caixa mais confiante; sem caixa = abstenção."""
    initialize_yolo_imports(repo)
    import torch
    from models.common import AutoShape, DetectMultiBackend
    backend = DetectMultiBackend(str(weights), device=torch.device(device))
    model = AutoShape(backend, verbose=False)
    model.conf = .25
    model.iou = .45
    # Mantém apenas as classes-alvo no baseline para não contar outras classes COCO.
    model.classes = [1, 15] if standard else [0, 1]
    df = pd.read_csv(Path(root) / 'manifest.csv')
    selected = df[df.split == split]
    for row in selected.head(2).itertuples():
        model(Image.open(Path(root)/'images'/split/row.filename).convert('RGB'), size=224)
    records, images = [], []
    for row in selected.itertuples():
        im = Image.open(Path(root)/'images'/split/row.filename).convert('RGB')
        times = []
        for _ in range(3):
            if device == 'mps':
                torch.mps.synchronize()
            elif device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = model(im, size=224)
            boxes = prediction.xyxy[0].cpu().numpy()
            predicted, confidence = -1, 0.0
            if len(boxes):
                best = boxes[np.argmax(boxes[:, 4])]
                predicted = COCO_MODEL_IDS[int(best[5])] if standard else int(best[5])
                confidence = float(best[4])
            if device == 'mps':
                torch.mps.synchronize()
            elif device.startswith('cuda'):
                torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        records.append(dict(filename=row.filename, true=int(row.class_id), predicted=predicted,
                            confidence=confidence, latency_ms=float(np.median(times))))
        annotated = im.copy()
        draw = ImageDraw.Draw(annotated)
        for x1, y1, x2, y2, conf, cls in boxes:
            c = COCO_MODEL_IDS[int(cls)] if standard else int(cls)
            draw.rectangle((x1,y1,x2,y2), outline='#16d99e', width=4)
            draw.text((x1, max(0,y1-15)), f'{NAMES[c]} {conf:.2f}', fill='#16d99e')
        images.append((row.filename, annotated))
    return records, images


def classification_metrics(records, model_name):
    """Abstenções são erros; matriz 2 x 3 inclui a coluna sem detecção."""
    from sklearn.metrics import accuracy_score, f1_score
    truth = [r['true'] for r in records]
    pred = [r['predicted'] for r in records]
    cm = np.zeros((2,3), dtype=int)
    for a,b in zip(truth,pred):
        cm[a, b if b >= 0 else 2] += 1
    return dict(model=model_name, accuracy=accuracy_score(truth,pred),
                error_rate=1-accuracy_score(truth,pred),
                macro_f1=f1_score(truth,pred,labels=[0,1],average='macro',zero_division=0),
                abstentions=sum(p == -1 for p in pred),
                latency_median_ms=float(np.median([r['latency_ms'] for r in records])),
                n_images=len(records), confusion_matrix=cm.tolist())


def make_cnn():
    """CNN pequena inicializada aleatoriamente, sem pesos pré-treinados."""
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(), nn.Dropout(.3),
                         nn.Linear(64*4*4,64), nn.ReLU(), nn.Linear(64,2))


def image_loader(root, split, training=False, batch_size=8):
    import torch
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    class Photos(Dataset):
        def __init__(self):
            df = pd.read_csv(Path(root)/'manifest.csv')
            self.rows = list(df[df.split==split].itertuples())
            operations = [transforms.Resize((160,160))]
            if training:
                operations += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(.15,.15,.15,.05)]
            self.transform = transforms.Compose(operations+[transforms.ToTensor(),
                transforms.Normalize([.5]*3,[.5]*3)])
        def __len__(self):
            return len(self.rows)
        def __getitem__(self,i):
            row = self.rows[i]
            im = Image.open(Path(root)/'images'/split/row.filename).convert('RGB')
            return self.transform(im), int(row.class_id), row.filename
    return DataLoader(Photos(), batch_size=batch_size, shuffle=training, num_workers=0,
                      generator=torch.Generator().manual_seed(SEED))


def train_cnn(root, results, epochs=60, device='cpu'):
    import torch
    import torch.nn as nn
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = make_cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    output = Path(results)/'cnn'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'execution.json').exists() and (output/'best_cnn.pt').exists():
        return json.loads((output/'execution.json').read_text())
    train, val = image_loader(root,'train',True), image_loader(root,'val')
    history, best = [], float('inf')
    started = time.perf_counter()
    for epoch in range(epochs):
        metrics = {}
        for split, loader in [('train',train),('val',val)]:
            model.train(split=='train')
            loss_total, hits, n = 0.,0,0
            with torch.set_grad_enabled(split=='train'):
                for x,y,_ in loader:
                    x,y = x.to(device),y.to(device)
                    logits = model(x)
                    loss = criterion(logits,y)
                    if split=='train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    loss_total += loss.item()*len(y)
                    hits += int((logits.argmax(1)==y).sum())
                    n += len(y)
            metrics[f'{split}_loss'] = loss_total/n
            metrics[f'{split}_accuracy'] = hits/n
        history.append(dict(epoch=epoch+1,**metrics))
        if metrics['val_loss'] < best:
            best = metrics['val_loss']
            torch.save(model.state_dict(),output/'best_cnn.pt')
        if (epoch+1)%10 == 0:
            print(f'CNN {epoch+1}/{epochs} | val_loss={metrics["val_loss"]:.4f}',flush=True)
    pd.DataFrame(history).to_csv(output/'history.csv',index=False)
    meta = dict(model='cnn_zero',epochs=epochs,train_seconds=time.perf_counter()-started,device=device,
                best_epoch=int(pd.DataFrame(history).val_loss.idxmin())+1)
    (output/'execution.json').write_text(json.dumps(meta,indent=2))
    return meta


def cnn_predictions(root, results, device='cpu', split='test'):
    import torch
    model = make_cnn().to(device)
    model.load_state_dict(torch.load(Path(results)/'cnn/best_cnn.pt',map_location=device,weights_only=True))
    model.eval()
    loader = image_loader(root,split,batch_size=1)
    records = []
    with torch.inference_mode():
        for _ in range(2):
            model(next(iter(loader))[0].to(device))
        for x,y,filenames in loader:
            times = []
            # Mesmo protocolo: input RGB decodificado -> pré-processamento -> decisão.
            for _ in range(3):
                from torchvision import transforms
                original = Image.open(Path(root)/'images'/split/filenames[0]).convert('RGB')
                transform = transforms.Compose([transforms.Resize((160,160)),transforms.ToTensor(),
                                                 transforms.Normalize([.5]*3,[.5]*3)])
                if device == 'mps':
                    torch.mps.synchronize()
                elif device.startswith('cuda'):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                logits = model(transform(original).unsqueeze(0).to(device))
                probabilities = logits.softmax(1)
                confidence,pred = probabilities.max(1)
                confidence,pred = float(confidence.item()),int(pred.item())
                if device == 'mps':
                    torch.mps.synchronize()
                elif device.startswith('cuda'):
                    torch.cuda.synchronize()
                times.append((time.perf_counter()-started)*1000)
            records.append(dict(filename=filenames[0],true=int(y.item()),predicted=pred,
                                confidence=confidence,latency_ms=float(np.median(times))))
    return records


def save_evidence(results, name, records, images=None):
    output = Path(results)/'predictions'/name
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(output/'predictions.csv',index=False)
    if images:
        for filename, image in images:
            image.save(output/filename)
    metric = classification_metrics(records,name)
    (output/'metrics.json').write_text(json.dumps(metric,indent=2))
    return metric


def create_report(results, detection_records, classification_records, training_records):
    """Relatório só usa resultados calculados; não pressupõe melhoria com mais épocas."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = Path(results)
    det, cls, training = map(pd.DataFrame,[detection_records,classification_records,training_records])
    det.to_csv(results/'detection_comparison.csv',index=False)
    cls.drop(columns=['confusion_matrix']).to_csv(results/'classification_comparison.csv',index=False)
    training.to_csv(results/'training_comparison.csv',index=False)
    validation = det[(det.split=='val') & det.model.isin(['yolo_30ep', 'yolo_60ep'])]
    winner = validation.sort_values(['map50_95','model'],ascending=[False,True]).iloc[0]['model']
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for epochs in [30,60]:
        history = pd.read_csv(results/f'train/yolo_{epochs}ep/results.csv')
        history.columns = history.columns.str.strip()
        axes[0].plot(np.arange(1,len(history)+1),history['metrics/mAP_0.5:0.95'],label=f'{epochs} épocas')
        total_loss = history['val/box_loss']+history['val/obj_loss']+history['val/cls_loss']
        axes[1].plot(np.arange(1,len(history)+1),total_loss,label=f'{epochs} épocas')
    axes[0].set(title='mAP50–95 na validação',xlabel='Época',ylim=(0,1))
    axes[1].set(title='Soma das perdas na validação',xlabel='Época')
    axes[2].bar(cls.model,cls.accuracy,color=['#238f77','#44ad9c','#4865bd','#d58a41'])
    axes[2].set(title='Acurácia por imagem (teste)',ylim=(0,1))
    axes[2].tick_params(axis='x',rotation=25)
    for ax in axes[:2]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(results/'comparison.png',dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,len(cls),figsize=(16,4))
    for ax,row in zip(axes,cls.itertuples()):
        matrix = np.asarray(row.confusion_matrix)
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=4)
        ax.set_xticks(range(3),NAMES+['sem detecção'],rotation=30)
        ax.set_yticks(range(2),NAMES)
        ax.set(title=row.model,xlabel='Predita',ylabel='Real')
        for i in range(2):
            for j in range(3):
                ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    fig.tight_layout()
    fig.savefig(results/'confusion_matrices.png',dpi=180)
    plt.close(fig)
    d30 = validation[validation.model=='yolo_30ep'].iloc[0]
    d60 = validation[validation.model=='yolo_60ep'].iloc[0]
    delta = d60.map50_95-d30.map50_95
    note = 'aumentou' if delta>0 else 'diminuiu' if delta<0 else 'não alterou'
    test = det[det.split=='test'].set_index('model')
    scores = cls.set_index('model')
    report = f'''# Resultados da execução — FarmTech Fase 6

## Protocolo
80 imagens: 40 bicicletas e 40 gatos; 64 treino, 8 validação e 8 teste.
YOLOv5n v7.0: duas inicializações iguais, 30 e 60 épocas, imagens 224 px,
batch 4 e semente 42. CNN do zero: 60 épocas, imagens 160 px.
Seleção dos pesos por validação. O teste é usado apenas para avaliação final.

## Detecção
{det.round(4).to_markdown(index=False)}

A passagem de 30 para 60 épocas **{note}** a mAP50–95 de validação
em {delta:+.4f}. A configuração selecionada pela validação é **{winner}**.
Mais épocas não garantem desempenho melhor: compare as curvas de perda,
o pico da mAP e a época do checkpoint escolhido. Perdas de box/obj/cls
são acompanhadas no treino; não são percentuais de erro e não são
comparáveis diretamente à entropia cruzada da CNN.

No teste, a YOLO padrão obteve mAP50–95 **{test.loc['yolo_padrao','map50_95']:.4f}**,
enquanto a configuração customizada selecionada obteve
**{test.loc[winner,'map50_95']:.4f}**. Nesta amostra, customizar a rede
não assegurou melhor localização do que utilizar pesos padrão. O detector pode
acertar a classe da foto e, ainda assim, perder instâncias ou detectar caixas falsas.

## Classificação comum às abordagens
{cls.drop(columns=['confusion_matrix']).round(4).to_markdown(index=False)}

O detector recebe como predição da imagem a classe da caixa mais confiante
entre as duas classes-alvo, com confiança mínima 0,25. Ausência de caixa
é abstenção, contada como erro. A CNN fornece diretamente uma das duas classes.
Esta comparação avalia a identificação por imagem; somente YOLO localiza objetos.

A CNN acertou **{round(scores.loc['cnn_zero','accuracy']*8)}/8** imagens;
o detector customizado escolhido acertou **{round(scores.loc[winner,'accuracy']*8)}/8**.
O treino de 30 épocas teve **{int(scores.loc['yolo_30ep','abstentions'])} abstenções**
com limiar 0,25. A mAP usa confiança mínima 0,001 para construir a curva PR;
portanto uma mAP acima de zero pode coexistir com várias abstenções no limiar
operacional. Mantivemos 0,25 fixo, sem ajustá-lo depois de observar o teste.

## Custo de treinamento
{training[['model','epochs','train_seconds','device','best_epoch']].round(3).to_markdown(index=False)}

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
'''
    (results/'RELATORIO_RESULTADOS.md').write_text(report)
    (results/'selection.json').write_text(json.dumps(dict(selected_model=winner,criterion='validation mAP50-95'),indent=2))
    return report
