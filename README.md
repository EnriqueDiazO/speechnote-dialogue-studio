# SpeechNote Dialogue Studio

Estudio local en Streamlit para escribir diálogos con varios hablantes, elegir Speech Note o
Qwen3-TTS por personaje o intervención, sintetizar y ensamblar un WAV maestro. El texto y el
audio permanecen en el equipo. Speech Note se controla por la CLI de su Flatpak y Qwen se ejecuta
en un servicio local aislado que sólo escucha en `127.0.0.1`.

## Requisitos

- Python 3.10 o posterior.
- Speech Note instalado como Flatpak y abierto durante la síntesis.
- En Speech Note: **Ajustes → Permitir aplicaciones externas para invocar acciones**.
- Al menos una voz TTS descargada en Speech Note.
- Para Qwen: el entorno separado `/home/enriquedo/PersonalProjects/qwen/.venv-qwen`, CUDA y una
  GPU compatible con BF16. La aplicación principal no importa Torch ni `qwen_tts`.
- `ffmpeg` y `ffprobe`; el WAV funciona sin la exportación MP3, pero la normalización necesita
  ambas herramientas si una voz produce un formato distinto del maestro.

Las voces iniciales son:

- Profesor: `es_piper_mx_claude_high`.
- Estudiante: `es_piper_es_sharvard_medium_1`.

La aplicación no descarga voces ni modifica la configuración de Speech Note.

## Instalación y ejecución

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
make run
```

Abre <http://127.0.0.1:8510>. Los atajos disponibles son:

```bash
make install   # crea .venv e instala app + herramientas de desarrollo
make run       # Streamlit en 127.0.0.1:8510
make test      # pruebas sin invocar Speech Note real
make lint      # Ruff
make doctor    # diagnóstico de Flatpak, Speech Note, voces, FFmpeg y Música
make qwen-status
make qwen-start
make qwen-unload  # libera el modelo de VRAM sin detener el servicio
make qwen-stop
```

## Flujo de trabajo

1. Abre Speech Note y habilita la invocación externa, o inicia el backend Qwen desde la UI.
2. Carga el proyecto de ejemplo o crea un proyecto nuevo.
3. Elige proveedor, voz e idioma por personaje. Una intervención puede tener un override propio.
4. Pulsa **Generar** en una tarjeta o **Generar pendientes**. Editar el texto, el hablante o su
   voz marca el audio como desactualizado, pero conserva la toma anterior.
5. Cuando todas estén listas, pulsa **Construir diálogo**. El master usa PCM de 16 bits,
   48 kHz, mono, y la pausa configurada entre intervenciones.
6. Escucha el resultado, descarga WAV, crea el MP3 opcional o exporta el proyecto ZIP portable.
7. Guarda para reabrir el proyecto desde la barra lateral.

No cierres Speech Note mientras una voz Piper está trabajando. Las síntesis se ejecutan de forma
secuencial; nunca se lanzan dos a la vez, incluso al comparar varias voces Qwen.

El modelo Qwen instalado es `Qwen3-TTS-12Hz-0.6B-CustomVoice`. Permite nueve voces, once idiomas
y controles de sampling. Aunque la API pública acepta `instruct`, la implementación 0.1.1 fuerza
ese valor a `None` para el tamaño `0b6`; por eso la UI no muestra emoción, estilo ni instrucciones
como controles activos. Tampoco anuncia VoiceDesign o clonación. Consulta
[docs/QWEN_TTS_BACKEND.md](docs/QWEN_TTS_BACKEND.md).

## Datos locales

La raíz se descubre con `xdg-user-dir MUSIC`, no se asume `~/Music`:

```text
<Música>/SpeechNote Dialogue Studio/
├── projects/<slug>-<id-corto>/
│   ├── project.json
│   ├── manifest.json
│   ├── audio/raw/
│   ├── audio/normalized/
│   └── exports/
├── temporary/qwen-previews/
└── runtime/                  # PID, log y lock de arranque del servicio; nunca del proyecto
```

Los JSON sólo contienen rutas relativas. El repositorio ignora audio, ZIP, entornos virtuales y
carpetas de trabajo. La aplicación no borra proyectos ni tomas anteriores automáticamente; sólo
reemplaza de forma segura los JSON del proyecto que el usuario ya abrió o creó.

Consulta [docs/PROJECT_FORMAT.md](docs/PROJECT_FORMAT.md) para el esquema portable y el contenido
del ZIP.

## Diagnóstico manual

```bash
flatpak info net.mkiol.SpeechNote
flatpak run net.mkiol.SpeechNote --print-available-models tts
flatpak run net.mkiol.SpeechNote --print-active-model tts
ffmpeg -version
ffprobe -version
xdg-user-dir MUSIC
```

Prueba manual mínima de Speech Note, eligiendo una salida nueva dentro de la carpeta Música:

```bash
flatpak run net.mkiol.SpeechNote \
  --action start-reading-text \
  --id es_piper_mx_claude_high \
  --text "Prueba breve del estudio de diálogo." \
  --output-file "/ruta/dentro/de/Música/prueba-nueva.wav"
```

Si aparece `Action invocation is not enabled in settings`, abre Speech Note y activa **Ajustes →
Permitir aplicaciones externas para invocar acciones**. Si no se crea el archivo, comprueba que
la aplicación esté abierta, que la voz esté descargada y que Flatpak tenga acceso a Música.

### Síntesis interrumpida

Una síntesis activa se coordina únicamente en memoria y nunca se guarda como lock del proyecto.
Si Streamlit o Speech Note se interrumpen, al volver a abrir el proyecto aparecerá **Recuperar
síntesis interrumpida** sólo cuando exista una inconsistencia. La recuperación:

- adopta como `ready` un WAV válido que corresponde a la intervención;
- marca como `stale` una salida ausente para permitir regenerarla;
- conserva una salida inválida bajo `audio/recovery/` con sufijo `.partial`;
- no borra tomas válidas, texto, hablantes ni orden.

`make doctor` informa cantidades de intervenciones interrumpidas, WAV recuperables, archivos
parciales y locks persistentes sin mostrar el contenido del guion. Funciona aunque Speech Note
esté cerrado.

## Alcance

Incluye edición multi-hablante, dos proveedores combinables, overrides, galería Qwen, persistencia
JSON, generación/regeneración individual y por lote, reproductores, normalización, WAV maestro,
MP3 y ZIP. Quedan fuera VoiceDesign, clonación, música, efectos, transcripción, diarización, nube,
bases de datos, edición de onda, reducción de ruido e IA generadora de guiones.
