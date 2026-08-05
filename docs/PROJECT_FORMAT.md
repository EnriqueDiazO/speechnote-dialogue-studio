# Formato de proyecto

SpeechNote Dialogue Studio usa `schema_version: 1`. El archivo `project.json` está serializado en
UTF-8, con claves ordenadas e indentación de dos espacios. Un lector debe rechazar versiones
futuras que no entienda y puede ignorar campos opcionales conocidos ausentes.

## Modelo

El objeto raíz contiene:

- `project_id`: UUID estable; el slug de carpeta es sólo descriptivo.
- `title`, `description`, `language` (`es-MX` por defecto).
- `pause_ms`: entero de 0 a 5000; por defecto 650.
- `speakers`: uno o más perfiles.
- `utterances`: intervenciones en orden consecutivo desde 1.
- `created_at`, `updated_at`: marcas ISO 8601.

Cada perfil de hablante contiene `speaker_id`, `name`, `model_id`, `model_label`, `color_key` y
`enabled`. Ningún personaje está fijado en el modelo; Profesor y Estudiante son sólo los perfiles
iniciales.

Cada intervención contiene `utterance_id`, `order`, `speaker_id`, `text`,
`audio_relative_path`, `duration_seconds`, `sha256`, `status`, `error_message`, `created_at` y
`updated_at`. Los estados válidos son `draft`, `generating`, `ready`, `error` y `stale`.

`generating` se admite al leer proyectos antiguos, pero representa exclusivamente un instante de
ejecución. Al guardar, se serializa como `stale` recuperable; identificadores como `busy`,
`active_synthesis_id`, `generation_in_progress`, tokens de sesión y locks nunca forman parte de
`project.json`.

`audio_relative_path` siempre usa una ruta POSIX relativa al directorio del proyecto, por ejemplo:

```text
audio/normalized/001-550e8400-e29b-41d4-a716-446655440000-a1b2c3d4e5.wav
```

No se admiten rutas absolutas, componentes `..`, symlinks ni bytes de audio dentro del JSON.

## Audio

Los archivos de `audio/raw/` son las salidas inalteradas de Speech Note. Los segmentos de
`audio/normalized/` usan WAV PCM signed 16-bit little endian, 48000 Hz y un canal. El master se
construye con `wave` sobre esos segmentos e inserta silencio sólo entre intervenciones.

La aplicación conserva tomas anteriores al regenerar y calcula SHA-256 de cada segmento y de los
exports. El MP3 opcional usa `libmp3lame`, 192 kbps, 48000 Hz y mono.

Un proyecto antiguo que contenga `generating`, o el mensaje histórico de concurrencia, puede
recuperarse sin cambiar el esquema. Los WAV válidos se verifican por identificador de intervención,
cabecera RIFF/WAVE, ffprobe, duración positiva y SHA-256. Los archivos inválidos se preservan con
sufijo `.partial`; no se reproducen ni se eliminan.

## ZIP portable

```text
speech-dialogue-project/
├── project.json
├── manifest.json
├── README.md
├── audio/
│   ├── segments/
│   ├── dialogue.wav
│   └── dialogue.mp3       # sólo si se generó
└── script/
    ├── dialogue.txt
    └── dialogue.md
```

El `project.json` interno remapea los segmentos a `audio/segments/`. `manifest.json` registra el
formato, versión, proyecto, voces, intervenciones, pausa y la lista de archivos. Cada entrada de
archivo incluye `path`, `role`, `size`, `sha256`, `media_type` y `duration_seconds` cuando aplica.
El orden y las marcas internas del ZIP son deterministas; no se incluyen modelos de voz ni
archivos externos al proyecto.
