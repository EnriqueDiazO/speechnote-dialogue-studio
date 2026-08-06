# Backend Qwen3-TTS

## Alcance verificado

La integración usa exclusivamente `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` con `qwen-tts 0.1.1`.
El entorno probado es Python 3.12.2, Torch y Torchaudio 2.7.1+cu118, CUDA, una RTX 3060 Ti de
8 GiB, `torch.bfloat16` y atención `sdpa`. No requiere ni instala `flash-attn`; tampoco instala el
modelo 1.7B o modifica el driver.

La API y el modelo cargado anuncian estas capacidades:

- `supports_instruct = false`
- `supports_voice_design = false`
- `supports_voice_cloning = false`
- `supports_sampling_controls = true`
- `supports_speaker_selection = true`
- `supports_language_selection = true`

La firma de `generate_custom_voice` contiene `instruct`, pero la implementación instalada fuerza
`instruct = None` cuando `tts_model_size == "0b6"`. Dialogue Studio no construye ni envía ese
campo para este modelo y oculta los controles emocionales. Los campos opcionales del dominio
permiten incorporar en el futuro un modelo que sí anuncie esa capacidad sin fingirla hoy.

## Aislamiento y arquitectura

Streamlit no importa Torch, Torchaudio o `qwen_tts`. `QwenBackendManager` resuelve el intérprete
externo e inicia otro proceso con:

```text
<ruta-qwen>/.venv-qwen/bin/python \
  -m dialogue_studio.qwen_service
```

Ese proceso es un controller que no importa Torch ni Qwen. Escucha sólo en `127.0.0.1:8765`,
ejecuta un preflight fail-closed y crea como máximo un proceso hijo
`dialogue_studio.qwen_worker`. Sólo el worker importa Torch, carga el modelo y toca CUDA. Expone:

- `GET /health`: controller, cola, worker, modelo, política y último preflight.
- `GET /capabilities`: voces, idiomas y capacidades reales.
- `GET /preflight`: comprobación actual de NVIDIA, CUDA, BF16, sesión y eventos del kernel.
- `GET /diagnostic`: diagnóstico exportable sin texto, audio ni rutas personales completas.
- `POST /synthesize`: encola una síntesis estrictamente secuencial con salida WAV atómica.
- `POST /unload`: descarga el modelo dentro del worker.
- `POST /worker/stop`: termina el worker y libera toda su VRAM.
- `POST /cancel`: cancela pendientes y permite terminar el trabajo actual de forma controlada.
- `POST /shutdown`: detiene controller y worker.

El worker se crea mediante `subprocess` (spawn del sistema operativo), nunca mediante `fork`
después de inicializar CUDA. El modelo se carga perezosamente y los trabajos esperan en una cola
única. Los timeouts predeterminados son 30 s para arranque, 180 s para carga y 300 s para
síntesis. A los 120 s inactivo se descarga el modelo y a los 300 s se termina el worker. PID,
cola, locks y peticiones activas nunca entran en `project.json`.

La salida se escribe primero como un nombre aleatorio `.partial` dentro de la raíz controlada. Se
valida RIFF/WAVE y sólo entonces `os.replace` la publica atómicamente. Una salida parcial de un
fallo se conserva para diagnóstico. Los errores HTTP tienen código, mensaje y atributo
`retryable`; un `gpu_busy` se traduce a estado transitorio y no se persiste como error del audio.

## Configuración

Variables admitidas y defaults:

```text
QWEN_TTS_PYTHON=/absolute/path/to/qwen/.venv-qwen/bin/python
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
QWEN_TTS_HOST=127.0.0.1
QWEN_TTS_PORT=8765
QWEN_TTS_DEVICE=cuda:0
QWEN_TTS_DTYPE=bfloat16
QWEN_TTS_ATTN=sdpa
QWEN_TTS_TIMEOUT=600
```

`QWEN_TTS_PYTHON` es opcional. La resolución usa, en orden, la variable de entorno, una ruta
inyectada explícitamente en `QwenClientConfig` y el proyecto hermano
`../qwen/.venv-qwen/bin/python`, derivado de la ubicación real de este repositorio sin depender
del usuario ni de su carpeta personal. La ruta seleccionada debe existir, ser un archivo
ejecutable, ejecutar Python e importar `qwen_tts`. La validación ocurre en un subproceso, no
sintetiza audio y no carga ni descarga ningún modelo.

El entorno Qwen, la instalación de `qwen-tts`, los pesos y la caché de Hugging Face viven fuera de
Dialogue Studio. Este repositorio no crea ni administra copias de esos recursos. Para ver la ruta
efectiva, su origen (`environment`, `configured` o `sibling-discovery`) y el resultado de las
comprobaciones:

```bash
make qwen-runtime
```

Host remoto, `float16`, otra atención y modelos distintos se rechazan. BF16 se mantiene
porque fue validado en la GPU instalada y evita los problemas NaN observables con precisión
inadecuada. La semilla se aplica dentro de `torch.random.fork_rng`, que restaura el estado del RNG
CPU/CUDA aunque la generación falle.

El modo CPU de emergencia sólo admite `cpu`, `float32` y `sdpa`, exige política y confirmación
explícitas y nunca sustituye automáticamente un trabajo CUDA fallido. La política y los límites
de seguridad se documentan en [QWEN_GPU_DISPLAY_SAFETY.md](QWEN_GPU_DISPLAY_SAFETY.md).

Opciones expuestas, todas confirmadas en `_merge_generate_kwargs` y en una síntesis CUDA real:

| Opción | Rango de la UI | Default |
|---|---:|---:|
| `seed` | 0–4294967295 | 0 |
| `max_new_tokens` | 64–8192 | 8192 |
| `temperature` | 0.1–2.0 | 0.9 |
| `top_p` | 0.1–1.0 | 1.0 |
| `top_k` | 1–200 | 50 |
| `repetition_penalty` | 0.8–2.0 | 1.05 |

## Voces e idiomas

`/capabilities` usa `model.get_supported_speakers()` y `model.get_supported_languages()` cuando
el modelo está cargado. Antes de la primera carga usa el catálogo verificado localmente.

Voces: Aiden, Dylan, Eric, Ono Anna, Ryan, Serena, Sohee, Uncle Fu y Vivian.

Idiomas: Automático, Chino, Inglés, Francés, Alemán, Italiano, Japonés, Coreano, Portugués, Ruso y
Español. Los IDs persistidos son minúsculos; los proyectos en español seleccionan `spanish` al
cambiar un personaje a Qwen.

## Uso en la interfaz

Cada personaje elige proveedor, voz e idioma. En Qwen aparece un expander con los controles de
sampling y restauración de defaults; no aparece un constructor emocional. Cada intervención puede
activar un override propio de proveedor, voz, idioma y diferencias de sampling.

**Probar voz** crea un WAV temporal sin modificar la intervención. **Explorar voces Qwen** permite
editar un texto, seleccionar varias voces, generarlas secuencialmente, reproducirlas, comparar
duración/tiempo y asignar una al personaje. La caché se direcciona por la huella de modelo, texto,
voz, idioma y sampling. **Limpiar previews** sólo borra WAV dentro de
`temporary/qwen-previews/`; no toca proyectos ni tomas definitivas.

El panel permite ejecutar preflight, iniciar de forma segura, descargar el modelo, detener el
worker, detener el servicio y descargar un diagnóstico. Muestra política, sesión, driver,
temperatura, uso, VRAM, procesos, Xid, worker, cola, modelo y tiempo inactivo.

## Audio y diálogos mixtos

Una intervención resuelve primero personaje más override. Speech Note conserva su adaptador y
Qwen recibe únicamente `text`, `speaker`, `language` y sampling. El servicio produce WAV PCM
16-bit mono a 24000 Hz; el cliente vuelve a validarlo con ffprobe. El normalizador existente crea
PCM 16-bit mono a 48000 Hz antes de actualizar duración, hash, huella y `ready`.

El master sólo consume segmentos normalizados, por lo que puede mezclar Speech Note y Qwen sin
una ruta especial. Un fallo conserva audios anteriores, deja la intervención regenerable y limpia
el coordinador en memoria. Reiniciar Streamlit o el backend no reconstruye locks desde el JSON.

## Operación y diagnóstico

```bash
make qwen-runtime
make qwen-status
make qwen-start
make qwen-unload
make qwen-stop
```

Los pesos permanecen en la caché normal de Hugging Face; descargar el modelo de VRAM no borra esa
caché. Si CUDA no está disponible o BF16 no está soportado, el servicio entra en `error` y Speech
Note sigue editable y operativo. Ante OOM, detén otras cargas, usa **Descargar modelo** y vuelve a
ejecutar el preflight; no cambies a FP16. Los logs locales del controller y worker están bajo
`<Música>/SpeechNote Dialogue Studio/runtime/`.

## Pruebas

La suite normal usa modelos y clientes falsos; no importa Qwen ni carga CUDA:

```bash
make test
make lint
```

La integración real es opt-in:

```bash
RUN_QWEN_REAL=1 .venv/bin/python -m pytest -q tests/test_qwen_real.py -s
```

Genera Serena, Vivian y Ryan con el mismo texto en `spanish`, comprueba carga única, hashes
distintos, PCM 24 kHz, ausencia de error/NaN, normalización 48 kHz, master mixto, MP3, recarga de
JSON y reinicio sin PID o lock persistente.
