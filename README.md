# Model Loader

Aplicação **desktop** (PySide6) para **detecção de objetos** com suporte a múltiplos modelos (`.pt` e `.onnx`), rodando em:

- **Cascade**: respeita uma ordem de execução; cada estágio roda sobre recortes (crops) das detecções do estágio anterior.
- **Simultâneo**: todos os modelos rodam no frame inteiro e as detecções são combinadas.

Fontes suportadas: **imagem**, **vídeo** e **câmera**.

## Funcionalidades

- **Carregar/remover modelos** `.pt` (Ultralytics/Torch) e `.onnx` (ONNX Runtime)
- **Ordenar e selecionar** quais modelos participam da execução
- **Executar** em imagem (processamento único) ou em stream (vídeo/câmera)
- **Ajustar limiares**: `confidence` e `iou`
- **Preview** com caixas desenhadas e legenda por classe/modelo

## Requisitos

- **Python 3.10+** (recomendado)
- Dependências em `requirements.txt`:
  - `ultralytics`, `torch`, `onnxruntime`, `opencv-python`, `PySide6`, `pyyaml`

> Observação: `torch`/`onnxruntime` podem variar por GPU/CPU. Em Windows, o pip pode instalar variantes diferentes conforme o ambiente.

## Como rodar (Windows)

No PowerShell, na raiz do projeto:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Como usar

1. Abra o app (`python main.py`)
2. Clique em **Carregar modelo** e selecione um `.pt` ou `.onnx`
3. Marque os modelos que deseja usar (checkbox)
4. Se estiver em **cascade**, ajuste a **ordem** conforme necessário
5. Escolha a **Fonte**:
   - `image`: selecione uma imagem
   - `video`: selecione um vídeo
   - `camera`: usa `camera:0` (pode trocar o índice se existir mais de uma câmera)
6. Ajuste **Execução** (`cascade`/`simultaneo`) e **Confiança**
7. Clique em **Iniciar**

## Configuração (`config.yaml`)

Existe um `config.yaml` com chaves para app/modelos/inferência (por padrão, thresholds e tipo de fonte). Atualmente, o app inicia com valores padrão (ex.: confiança 0.25) e a UI permite alterar durante a execução.

Arquivo: `config.yaml`

## Modelos

- Pasta padrão: `models/`
- O repositório inclui `models/.gitkeep` (sem pesos por padrão).

Se você quiser versionar pesos no Git, considere **Git LFS** (especialmente para `.pt` grandes).

## Build (PyInstaller)

O projeto já inclui `predict_system.spec`. Para gerar um executável:

```bash
pip install pyinstaller
pyinstaller .\predict_system.spec
```

Saídas típicas:

- `dist/` (build padrão)
- `build/` (artefatos intermediários)

> Dica: se você for distribuir, teste em uma máquina “limpa” para garantir que `torch/onnxruntime/cv2` foram empacotados corretamente.

## Estrutura do projeto

- `main.py`: entrypoint do app
- `src/ui/`: UI (janela principal, seletor de modelos, painel de preview)
- `src/core/`: carregamento de modelos e pipeline (cascade/simultâneo)
- `src/inputs/`: leitores de fonte (imagem/vídeo/câmera)
- `models/`: diretório recomendado para modelos

## Troubleshooting

- **Modelo não carrega**: confirme extensão `.pt`/`.onnx` e se o arquivo existe.
- **ONNX Runtime sem GPU**: é normal cair em CPU dependendo da instalação; verifique providers disponíveis.
- **Câmera não abre**: tente trocar `camera:0` para `camera:1` (ou feche apps que estejam usando a câmera).

## Licença

Defina a licença do projeto (ex.: MIT) adicionando um arquivo `LICENSE`.

