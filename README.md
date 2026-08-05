# SpeechNote Dialogue Studio

Estudio local en Streamlit para escribir diálogos con varios hablantes, asignar una voz
instalada de Speech Note a cada uno, sintetizar cada intervención y ensamblar un WAV maestro.
El texto y el audio permanecen en el equipo: la aplicación controla exclusivamente la CLI del
Flatpak `net.mkiol.SpeechNote` y no incluye un motor TTS propio.

## Requisitos

- Python 3.10 o posterior.
- Speech Note instalado como Flatpak y abierto durante la síntesis.
- En Speech Note: **Ajustes → Permitir aplicaciones externas para invocar acciones**.
- Al menos una voz TTS descargada en Speech Note.
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
```

## Flujo de trabajo

1. Abre Speech Note y habilita la invocación externa.
2. Carga el proyecto de ejemplo o crea un proyecto nuevo.
3. Ajusta hablantes y voces; añade, edita, duplica o reordena intervenciones.
4. Pulsa **Generar** en una tarjeta o **Generar pendientes**. Editar el texto, el hablante o su
   voz marca el audio como desactualizado, pero conserva la toma anterior.
5. Cuando todas estén listas, pulsa **Construir diálogo**. El master usa PCM de 16 bits,
   48 kHz, mono, y la pausa configurada entre intervenciones.
6. Escucha el resultado, descarga WAV, crea el MP3 opcional o exporta el proyecto ZIP portable.
7. Guarda para reabrir el proyecto desde la barra lateral.

No cierres Speech Note mientras una voz está trabajando. Las síntesis se ejecutan de forma
secuencial; nunca se lanzan dos a la vez.

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
└── temporary/
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

## Alcance del MVP

Incluye edición multi-hablante, persistencia JSON, generación/regeneración individual y por lote,
reproductores, reordenamiento, normalización, WAV maestro, MP3 y ZIP. Quedan fuera clonación de
voz, música, efectos, transcripción, diarización, nube, bases de datos, edición de onda, reducción
de ruido, fórmulas habladas e IA generadora de guiones.
