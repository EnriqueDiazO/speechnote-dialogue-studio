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
- `pronunciation_profile`: idioma y políticas de matemáticas, siglas, números, unidades y
  puntuación.
- `pronunciation_rules`: sólo reglas creadas o modificadas por el usuario en este proyecto.
- `created_at`, `updated_at`: marcas ISO 8601.

Cada perfil de hablante contiene `speaker_id`, `name`, `model_id`, `model_label`, `color_key` y
`enabled`. `model_id` y `model_label` se conservan para lectores antiguos. El campo opcional `tts`
es la configuración vigente:

- `provider`: `speechnote` o `qwen`;
- `voice_id`, `voice_label`;
- `language`;
- `generation_options`: sólo sampling numérico soportado;
- `instruction_text`: reservado para un modelo futuro que anuncie `supports_instruct`; se omite
  en la configuración actual 0.6B.

Un proyecto antiguo sin `tts` se interpreta en memoria como `provider: speechnote`, usando
`model_id` como voz. Abrirlo no reescribe el archivo; los campos nuevos aparecen sólo cuando el
usuario guarda explícitamente.

Cada intervención contiene `utterance_id`, `order`, `speaker_id`, `text` y su alias portable
`written_text`,
`audio_relative_path`, `duration_seconds`, `sha256`, `status`, `error_message`, `created_at` y
`updated_at`. Puede contener `tts_override` y `audio_fingerprint`. El override admite proveedor,
voz, idioma, opciones de sampling e instrucción futura; sólo se guardan sus diferencias duraderas.
También puede contener `use_pronunciation_engine`, `manual_spoken_text_override`,
`utterance_rules`, `spoken_text`, hashes escrito/hablado/de reglas, versión del motor, reglas
aplicadas y warnings. La huella SHA-256 identifica texto escrito, texto hablado, perfil, reglas,
override, personaje, proveedor, modelo, voz, idioma, instrucción y opciones efectivas. Los estados
válidos son `draft`, `generating`, `ready`, `error` y `stale`.

`generating` se admite al leer proyectos antiguos, pero representa exclusivamente un instante de
ejecución. Al guardar, se serializa como `stale` recuperable; identificadores como `busy`,
`active_synthesis_id`, `generation_in_progress`, tokens de sesión y locks nunca forman parte de
`project.json`.

Cambiar cualquier entrada de la huella marca el audio `stale` sin borrar su toma anterior.
`audio_relative_path` siempre usa una ruta POSIX relativa al directorio del proyecto, por ejemplo:

```text
audio/normalized/001-550e8400-e29b-41d4-a716-446655440000-a1b2c3d4e5.wav
```

No se admiten rutas absolutas, componentes `..`, symlinks ni bytes de audio dentro del JSON.

## Audio

Los archivos de `audio/raw/` son las salidas inalteradas del proveedor: Speech Note o Qwen a
24000 Hz. Los segmentos de `audio/normalized/` usan WAV PCM signed 16-bit little endian, 48000 Hz
y un canal. El master se construye con `wave` sobre esos segmentos e inserta silencio sólo entre
intervenciones.

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
formato, versión, proyecto, perfil y reglas de pronunciación, voces, intervenciones, pausa y la
lista de archivos. Los guiones conservan el texto escrito; la metadata conserva también el texto
hablado derivado. Cada entrada de
archivo incluye `path`, `role`, `size`, `sha256`, `media_type` y `duration_seconds` cuando aplica.
El orden y las marcas internas del ZIP son deterministas; no se incluyen modelos de voz ni
archivos externos al proyecto.
