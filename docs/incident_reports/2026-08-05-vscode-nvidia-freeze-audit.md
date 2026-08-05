# Auditoría forense: congelamiento de VS Code y pantallas NVIDIA

Fecha del incidente: 2026-08-05
Ventana solicitada: 16:10:00–16:40:00 CST (`UTC-06:00`)
Ventana causal principal: arranque `-1`, 16:10:00–16:30:48
Fecha/hora de inicio de la auditoría: 2026-08-05T16:37:58-06:00
Modo: diagnóstico; no se aplicaron correcciones

## Resumen ejecutivo

La causa más probable es una **interacción del proceso GPU de VS Code/Electron con la sesión gráfica X11 híbrida Intel/NVIDIA (PRIME) y GNOME/Mutter**, con **confianza media**.

El hecho mejor demostrado es que un `gpu-process` de VS Code/Electron quedó colgado: después de volver a abrir VS Code a las 16:28:20, Crashpad generó a las 16:28:50 un minidump cuyas anotaciones identifican `ptype=gpu-process`, `process_type=gpu-process`, `gpu_watchdog_thread.cc` y `list-of-hung-threads=149043`. El propio volcado indica que ese proceso usaba ANGLE/OpenGL sobre Intel UHD Graphics 730/Mesa, no CUDA ni la NVIDIA.

La dimensión multipantalla también tiene evidencia: la sesión era X11 híbrida; Intel exponía `HDMI-2`, mientras `NVIDIA-G0` exponía las dos salidas `HDMI-1-0` y `HDMI-1-1`. A las 16:25:27 GNOME Shell registró que no podía actualizar las *stage views* de un `MetaWindowActorX11`. Esto coincide con una composición por monitor afectada y con el síntoma reportado de que las salidas NVIDIA se congelaron mientras el resto del sistema seguía vivo.

No se demostró un reinicio espontáneo del driver NVIDIA antes del congelamiento. El primer error NVIDIA dentro de la ventana fue a las 16:29:31, exactamente cuando se ordenó manualmente `sudo reboot`. Durante el apagado, Xorg/NVIDIA registró varios `WAIT`, `nvidia-modeset` no logró poner en reposo el motor de pantalla y finalmente apareció Xid 31 a las 16:30:47. Un reinicio manual previo de `display-manager` a las 11:49:24 había producido la misma secuencia y otro Xid 31; por eso estos errores prueban que el cierre de Xorg/NVIDIA se atasca, pero no prueban que el Xid 31 iniciara el incidente.

Qwen no estaba sintetizando en la ventana investigada según la evidencia disponible. Su log dejó de modificarse a las 16:02:52 y no contiene excepciones CUDA/OOM. Hubo un Xid 43 asociado a un proceso llamado `python` a las 15:09:06, pero ocurrió 71 minutos antes del inicio aproximado y antes de la creación del log Qwen actual (15:40:04); no hay datos conservados que permitan asignar ese PID a Qwen. Speech Note tampoco registró crash, error CUDA o timeout. No hubo OOM, swap saturada, disco lleno, error de I/O ni tarea bloqueada en el kernel.

## Alcance y restricciones

Se inspeccionaron:

- repositorio, integridad Git y pruebas que no requieren síntesis real;
- arranque anterior (`2565785cee964ed082a6f6e396d58df0`) y arranque posterior (`d186bac8308a465f8fd76398b9bd3891`);
- kernel, DRM, NVIDIA, Xorg, GDM, GNOME Shell y topología de pantallas;
- logs y Crashpad de VS Code/Electron, extension host y Codex;
- estado y log de Qwen; procesos/logs de Speech Note;
- memoria, swap, filesystem, I/O observable, procesos y OOM.

No se modificó código ni configuración, no se actualizó software, no se detuvo ningún proceso, no se reinició el sistema durante la auditoría y no se lanzó síntesis Qwen. El único archivo creado es este reporte.

## Estado inicial

### Repositorio

```text
Repositorio: /home/enriquedo/PersonalProjects/speechnote-dialogue-studio
git status --short: sin salida (limpio)
Rama: main
HEAD: 02be7e01d55907ed82753d0e4c4f5e8e503690e1
HEAD corto: 02be7e0 fix(qwen): remove transient startup lock file
```

Los ocho commits iniciales coincidieron con el estado declarado en el incidente; no había cambios locales.

### Arranque y reinicio

```text
Hora de auditoría inicial: 2026-08-05T16:37:58-06:00
uptime -s:              2026-08-05 16:31:06
who -b:                 2026-08-05 16:31
arranque anterior:      2026-08-05 11:45:47–16:30:48
arranque actual:        comenzó 2026-08-05 16:31:14
```

`last -x -F` confirma un apagado ordenado a las 16:30:48 y un nuevo arranque a las 16:31:14. No fue un reinicio espontáneo: journald conserva `sudo ... COMMAND=/usr/sbin/reboot` a las 16:29:31.

### Plataforma gráfica medida después del reinicio

```text
Kernel: 6.8.0-136-generic
GPU discreta: NVIDIA GeForce RTX 3060 Ti, 8192 MiB
Driver NVIDIA: 535.309.01
CUDA reportada por nvidia-smi: 12.2
GPU integrada: Intel UHD Graphics 730 (PCI 8086:4692)
Sesión: X11 (XDG_SESSION_TYPE=x11, DISPLAY=:1, sin WAYLAND_DISPLAY)
```

`nvidia-smi -q` no indicó que hiciera falta reset después del reinicio (`Reset Required: No`, `Drain and Reset Recommended: No`). En la muestra de las 16:38 sólo Xorg usaba NVIDIA (83 MiB); Qwen y Speech Note no estaban ejecutándose después de este arranque.

## Línea temporal

| Hora | Fuente | Evento | Interpretación prudente |
|---|---|---|---|
| 11:49:24 | `sudo`/systemd | Se ejecutó manualmente `systemctl restart display-manager`. | Precedente útil para interpretar los errores del teardown NVIDIA. |
| 11:49:27–11:50:40 | Xorg/kernel | `NVIDIA(GPU-0): WAIT`, timeouts de display engine y Xid 31 de Xorg. | La misma familia de errores aparece al desmontar Xorg de forma intencional. |
| 15:09:06 | kernel | `NVRM: Xid ... 43, pid=100746, name=python`. | Falla real de un canal GPU Python; el PID no puede atribuirse a Qwen con los logs conservados. |
| 15:40:04 | filesystem | Se crea `qwen-tts.log`. | Es posterior al Xid 43. |
| 16:02:52 | filesystem | Última modificación de `qwen-tts.log`. | No hay actividad registrada por Qwen durante el incidente. |
| 16:10:02 | Git | Commit `02be7e0`. | El repositorio quedó limpio antes del incidente. |
| 16:11:03 | systemd de usuario | Se inician dos scopes de Speech Note. | Coincidencia temporal; sin error asociado. |
| 16:20:57–16:20:58 | `Codex.log` | Refresco periódico remoto completado con `success=true`, `error=null`. | Codex/app-server seguía respondiendo al inicio aproximado de la ventana. |
| 16:24:15 | systemd de usuario | Se inicia GNOME Terminal. | El sistema y la sesión aún aceptaban acciones. |
| 16:24:19 | systemd de usuario | Se inicia un scope de Firefox. | Coincidencia temporal; no hay crash de Firefox. |
| 16:25:27 | GNOME Shell | Tres mensajes `Can't update stage views ... needs an allocation` para `MetaWindowGroup`, `MetaWindowActorX11` y `MetaSurfaceActorX11`. | Primer indicio gráfico contemporáneo; no identifica qué ventana originó el actor. |
| 16:26:58 | systemd de usuario | `gsd-media-keys` lanza una aplicación y un VTE. | El resto del sistema seguía operativo. |
| 16:27:45 | VS Code | El extension host PID 20702 sale con código 0. | Cierre limpio, no crash del extension host. |
| 16:28:20 | VS Code | Empieza una nueva sesión de logs de VS Code. | VS Code fue reabierto antes del reinicio del sistema. |
| 16:28:50 | Crashpad | Minidump de un `gpu-process`; watchdog e hilo GPU colgado. | Evidencia directa de bloqueo del proceso GPU de Electron. |
| 16:29:31 | `sudo` | Se ejecuta `/usr/sbin/reboot`. | Inicio explícito del apagado. |
| 16:29:31 | kernel | `nvidia-drm ... Failed to grab modeset ownership`. | Ocurre durante el teardown, no antes. |
| 16:29:33–16:29:49 | kernel | Seis `Idling display engine timed out`. | El motor de pantalla NVIDIA no se desmontó con normalidad. |
| 16:29:34–16:30:27 | Xorg | Nueve mensajes `NVIDIA(GPU-0): WAIT`. | Xorg espera a NVIDIA durante el apagado. |
| 16:30:47 | kernel | Xid 31/MMU fault, PID 9485 `Xorg`. | Falla tardía durante el apagado; no prueba el inicio del congelamiento. |
| 16:30:48 | systemd-shutdown | Sincroniza filesystem y envía SIGTERM. | Apagado ordenado. |
| 16:31:14 | journal | Comienza el nuevo arranque. | Recuperación mediante reinicio. |
| 16:32:38 | VS Code | Nueva sesión de logs posterior al reinicio. | VS Code vuelve a abrir. |

## NVIDIA, kernel, DRM y pantallas

### Evidencia a favor de participación del stack NVIDIA

- La topología era híbrida. `xrandr --listproviders` mostró `modesetting` (Intel, `Source Output`) y `NVIDIA-G0` (`Sink Output`).
- La salida Intel era `HDMI-2` y las dos salidas `HDMI-1-0`/`HDMI-1-1` pertenecían a NVIDIA. Esto respalda que el síntoma pudiera limitarse a las dos pantallas NVIDIA.
- Durante el apagado, el motor de pantalla NVIDIA produjo timeouts repetidos y Xorg produjo `WAIT` repetidos antes del Xid 31.
- El driver ya mostraba una conducta anómala durante el teardown del `display-manager` de las 11:49:24.

### Evidencia que limita esa atribución

- Entre 16:10:00 y 16:29:30 no hubo Xid, NVRM, DRM reset, `fallen off the bus`, lockup u OOM en el kernel.
- El Xid 31 ocurrió 76 segundos después de la orden manual de reinicio.
- El patrón de `WAIT`/timeout/Xid 31 ya había aparecido al reiniciar manualmente el display manager; puede ser una consecuencia del teardown en esta configuración.
- Después del reinicio, `nvidia-smi -q` no solicitó reset y no mostró errores ECC/remapeos disponibles para esta GPU de consumo.

### Xid 43 anterior

El único Xid no asociado al cierre de Xorg fue:

```text
2026-08-05T15:09:06-0600 kernel: NVRM: Xid (PCI:0000:01:00): 43,
pid=100746, name=python, Ch 00000020
```

Es evidencia de un fallo previo en un canal GPU de Python, pero no hay `_PID` de journald, coredump ni registro de ejecución que revele el comando. Como ocurrió antes de la creación del log Qwen inspeccionado y hubo actividad normal durante más de una hora después, se clasifica como antecedente relevante, no como causa demostrada del congelamiento de las 16:20–16:30.

## GNOME Shell, Mutter, Xorg y GDM

La sesión era GNOME sobre X11. No se encontró crash o segfault de GNOME Shell ni reinicio espontáneo de GDM dentro de la ventana. El evento relevante previo al reinicio fue:

```text
16:25:27 gnome-shell: Can't update stage views actor <MetaWindowGroup> ... needs an allocation
16:25:27 gnome-shell: Can't update stage views actor <MetaWindowActorX11> ... needs an allocation
16:25:27 gnome-shell: Can't update stage views actor <MetaSurfaceActorX11> ... needs an allocation
```

Esto demuestra un problema de composición/asignación de una ventana X11, pero el mensaje no conserva el título, PID o aplicación de esa ventana. Los errores Xorg/NVIDIA posteriores empiezan al apagar la sesión, por lo que GNOME/Mutter se clasifica como parte probable de la interacción o como víctima, no como causa aislada demostrada.

## VS Code, Electron y Codex

### VS Code/Electron

```text
VS Code: 1.103.1 (commit 360a4e4fd251bfce169a4ddf857c7d25d1ad40da)
Arquitectura: x64
Electron según el minidump: 37.2.3
Invocación observada después del reinicio: /usr/share/code/code .
```

No se observó `--disable-gpu`. Los logs de la sesión iniciada a las 13:10 no contienen cadenas de GPU crash, renderer unresponsive, OOM, fatal o segfault. El extension host PID 20702 terminó con código 0 a las 16:27:45.

Sin embargo, la sesión de VS Code iniciada a las 16:28:20 produjo este artefacto:

```text
~/.config/Code/Crashpad/completed/3d82cc16-3990-461b-a398-57fb5c2c3201.dmp
mtime: 2026-08-05 16:28:50.615 -0600
tamaño: 193904 bytes
```

Las cadenas/anotaciones del minidump incluyen:

```text
ptype = gpu-process
process_type = gpu-process
mode = GpuMain
DumpWithoutCrashing-file = gpu/ipc/service/gpu_watchdog_thread.cc
list-of-hung-threads = 149043
gpu-gl-renderer = ANGLE (Intel, Mesa Intel(R) UHD Graphics 730 ...)
gpu-gl-vendor = Google Inc. (Intel)
gpu-driver = 23.2.1
```

`minidump_stackwalk` no estaba instalado, así que no fue posible simbolizar la pila. Aun así, las anotaciones son evidencia directa de que el watchdog detectó un hilo del GPU process colgado. El volcado fue `DumpWithoutCrashing`; no debe describirse como segfault.

No hubo coredump de Code/Electron/Xorg/GNOME/NVIDIA en `coredumpctl`. No hubo otro minidump de Code dentro de la ventana: el artefacto de las 16:28:50 fue el único.

### Extensión Codex/OpenAI

```text
Extensión: openai.chatgpt 26.727.40816
Ruta: ~/.vscode/extensions/openai.chatgpt-26.727.40816-linux-x64
```

Datos relevantes:

- `Codex.log` registró 356 errores `ResizeObserver loop completed with undelivered notifications` entre 13:45:46 y 16:12:38. Esto demuestra churn/errores de layout en el webview, pero el último precede en unos 13 minutos al evento de GNOME.
- El renderer avisó de estado global grande de `openai.chatgpt`, aproximadamente 1.15 MiB. No hay OOM ni bloqueo derivado demostrado.
- A las 16:20:58 el refresco periódico de Codex terminó con `success=true` y `error=null`, evidencia de que app-server seguía vivo.
- Los errores ENOENT de las 16:12:38 se referían a una referencia temporal de Git ya inexistente; no son errores GPU.
- No hay evidencia de crash o bloqueo del extension host, y su salida a las 16:27:45 fue limpia.
- Los errores de API propuesta e incompatibilidades de extensiones reaparecen también en arranques sanos; no tienen correlación específica con el congelamiento.

Conclusión parcial: Codex pudo aportar carga de layout/renderizado, pero no hay evidencia para asignarle la causa raíz. El componente de VS Code que sí quedó documentado como colgado fue el `gpu-process` de Electron.

## Qwen3-TTS y Speech Note

### Qwen

Después del reinicio:

```text
make qwen-status: state=offline, ok=false, last_error=null
pgrep: sin proceso dialogue_studio.qwen_service
nvidia-smi: sólo Xorg, 87 MiB
```

El archivo `/home/enriquedo/Música/SpeechNote Dialogue Studio/runtime/qwen-tts.log` medía 1079 bytes, se creó a las 15:40:04 y dejó de modificarse a las 16:02:52. Sólo contiene:

- aviso de que `flash-attn` no está instalado;
- dos cargas de cuatro archivos completadas;
- mensajes de `pad_token_id`.

No contiene `CUDA`, `error`, `exception`, `OOM`, `shutdown` ni traceback. Tampoco contiene timestamps o cambios de estado, por lo que no permite reconstruir cada solicitud. El código carga el modelo de forma perezosa, serializa la generación y la prueba real añadida en `6c0fb79` termina deteniendo/reiniciando el servicio y verificando `state=idle`, `model_loaded=false`. El consumo posincidente suministrado por el operador (~130 MiB) es compatible con un servicio Python/PyTorch ocioso sin el modelo cargado, pero esto es una inferencia, no una muestra conservada por el log.

El commit `02be7e0` sólo elimina el lock transitorio de arranque después de lanzar el proceso, y lo elimina también si `Popen` falla. No cambia CUDA, carga de modelo, generación o drivers. No se encontró un lock/PID activo después del reinicio; `qwen-status` informó offline.

No se volvió a lanzar síntesis real durante esta auditoría.

### Speech Note

Speech Note inició scopes a las 16:11:03. Al apagarse, systemd indicó que uno de sus scopes había consumido 4 min 59.222 s de CPU acumulada. No se encontraron mensajes de CUDA, GPU, crash, hang o timeout de Speech Note entre 16:10 y 16:30. Después del reinicio no estaba ejecutándose durante la muestra. Se clasifica como coincidencia temporal sin evidencia causal.

## Memoria, swap, disco e I/O

Muestra posterior al reinicio:

```text
RAM: 31 GiB total, 5.2 GiB usada, 24 GiB disponible
Swap: 2 GiB, 0 B usada
Filesystem raíz: 1.8 TiB, 647 GiB usados, 1.1 TiB libres (38 %)
Inodos raíz: 5 % usados
vmstat (muestras activas): 88–92 % CPU idle, 0–1 % iowait, sin swap in/out
```

En el arranque del incidente y el actual no hubo:

- OOM del kernel o de `systemd-oomd`;
- proceso matado por falta de memoria;
- `hung task`, soft/hard lockup;
- filesystem read-only, `no space left` o error de I/O;
- proceso actual en estado `D` o CPU runaway sostenida.

`sar` estaba instalado, pero `/var/log/sysstat` no contenía `sa05`; no existe una serie histórica para medir RAM, run queue o I/O exactamente a las 16:25. Por eso la presión de recursos se descarta por ausencia de eventos y por el estado posterior, no mediante una medición contemporánea exacta.

Firefox abrió un scope a las 16:24:19 y no dejó crash report/minidump en la ventana. Sus mensajes `Exiting due to channel error` son de las 16:29:31, durante el apagado. Su uso elevado de CPU observado después del reinicio no puede extrapolarse al incidente.

## Integridad del repositorio

Antes de crear este documento:

- `git status --short`: limpio;
- `git diff --check`: sin salida;
- `git fsck --full`: sin salida;
- `python -m compileall -q dialogue_studio tests`: correcto;
- `ruff check .`: `All checks passed!`;
- `pytest -q`: `87 passed, 1 skipped in 4.77s`.

La prueba omitida es la validación real que requiere `RUN_QWEN_REAL=1` y GPU; no se habilitó para respetar la prohibición de regenerar audio.

## Matriz de hipótesis

La confianza de la última columna expresa cuánto respalda la evidencia que esa hipótesis explique el incidente, no sólo que el componente estuviera presente.

| Hipótesis | Evidencia a favor | Evidencia en contra | Confianza |
|---|---|---|---|
| Interacción VS Code/Electron + X11/PRIME + GNOME/NVIDIA | GPU process colgado; error de *stage views*; topología híbrida; sólo salidas NVIDIA reportadas como congeladas; teardown NVIDIA atascado. | No hay captura viva ni evento único que establezca el orden causal interno. | **Media** |
| Proceso GPU de VS Code/Electron | Minidump directo del watchdog a las 16:28:50, `ptype=gpu-process`, hilo colgado. | Ocurre tras reabrir VS Code; usaba Intel/Mesa y puede ser víctima del stack gráfico, no iniciador. | **Media** como causa; **Alta** como evento ocurrido |
| Bloqueo/reinicio del driver NVIDIA | Timeouts del display engine, Xorg `WAIT`, Xid 31; congelamiento limitado a salidas NVIDIA. | Todo empieza tras `sudo reboot`; no hay reset/fallen-off/Xid antes de 16:29:31. | **Media** como factor, **Baja** como reset iniciador |
| Error Xid/NVRM/DRM del kernel | Xid 31 de Xorg a las 16:30:47; Xid 43 de Python a las 15:09:06. | Xid 31 es tardío durante teardown; Xid 43 está separado 71 minutos y sin atribución. | **Baja** como causa inmediata |
| GNOME Shell/Xorg | Error de *stage views* a las 16:25:27; Xorg se atasca al cerrar. | Sin crash/segfault/reinicio espontáneo; actor no identifica aplicación. | **Media** como factor o víctima |
| Extensión Codex/OpenAI o extension host | 356 errores de `ResizeObserver`; estado global de extensión ~1.15 MiB. | Último error 16:12:38; refresco exitoso 16:20:58; extension host sale con código 0. | **Baja** |
| Qwen/CUDA | Xid 43 de un `python` a las 15:09:06; uso de CUDA en el entorno. | No se puede asignar el PID; log Qwen posterior; sin actividad desde 16:02:52 ni error CUDA/OOM en el incidente. | **Baja** |
| Speech Note | Estaba abierto y usa capacidades GPU. | Sin error, crash, timeout o actividad correlacionada; sólo cierre ordenado. | **Sin evidencia** |
| Presión de RAM/swap/I/O/disco | No hay evidencia positiva. | Sin OOM/hung task/I/O error; swap vacía y amplio espacio tras reinicio. | **Sin evidencia** |
| Firefox/Electron/UI saturada | Firefox abrió un scope a las 16:24:19, cerca del evento GNOME. | Sin crash report ni error antes del apagado; la carga exacta no se conservó. | **Baja** |

## Causa raíz probable y grado de confianza

**Causa raíz probable:** bloqueo en la frontera de renderizado/composición de **VS Code/Electron y GNOME/Mutter dentro de la sesión X11 híbrida Intel/NVIDIA**, con el `gpu-process` de Electron como componente directamente probado que quedó colgado y el camino de salidas NVIDIA como manifestación/factor del congelamiento multipantalla.

**Grado de confianza: Media.**

No se eleva a alta porque:

1. el minidump corresponde al VS Code reabierto a las 16:28:20, no necesariamente a la instancia que empezó a fallar;
2. el GPU process del dump renderizaba sobre Intel/Mesa;
3. no hay Xid/DRM/NVRM previo a la orden de reinicio;
4. no se conservó una captura viva de procesos, GPU, Xorg o Mutter durante el primer segundo del congelamiento.

### Clasificación causal

- **Causa probable:** interacción del proceso GPU de Electron con composición X11 híbrida/PRIME.
- **Factor contribuyente probable:** GNOME/Mutter por monitor y la ruta de presentación hacia las salidas NVIDIA.
- **Evento confirmado pero causalidad ambigua:** teardown anómalo NVIDIA/Xorg y Xid 31 durante el reinicio.
- **Coincidencias temporales:** Firefox y Speech Note abiertos.
- **Aspectos descartados por los logs:** OOM, swap saturada, disco lleno, I/O error, extension host crash, coredump del escritorio.
- **No comprobable:** identidad de la ventana `MetaWindowActorX11`, comando del PID Python 100746 y pila simbolizada del minidump.

## Datos ausentes

- telemetría histórica `sar`/sysstat del minuto del incidente;
- `nvidia-smi -q`, procesos y temperatura capturados antes de reiniciar;
- `nvidia-bug-report` obtenido mientras las salidas estaban congeladas;
- pila simbolizada del minidump de Electron (`minidump_stackwalk` no disponible);
- timestamps y transiciones `loading_model`/`generating`/`idle` en el log Qwen;
- comando completo o cgroup del PID 100746 del Xid 43;
- título/PID de la ventana asociada al `MetaWindowActorX11` de GNOME;
- métrica viva por monitor que distinga bloqueo del compositor de pérdida de señal.

## Recomendaciones ordenadas por riesgo

Estas medidas se proponen; **ninguna fue ejecutada**.

1. **Riesgo mínimo — captura:** preparar un script de diagnóstico que conserve journal, `nvidia-smi`, topología, procesos, memoria y Crashpad inmediatamente, antes de reiniciar.
2. **Riesgo mínimo — observabilidad:** añadir timestamps, PID, estado y duración a Qwen para poder excluir o confirmar una síntesis sin inferencias.
3. **Riesgo bajo — comparación A/B:** si se repite, abrir una sesión separada de VS Code con `--disable-gpu` y comparar estabilidad, sin convertirlo todavía en configuración permanente.
4. **Riesgo bajo — aislamiento de extensión:** comparar una sesión de VS Code sin extensiones o sin Codex para separar el webview/layout del GPU process base.
5. **Riesgo bajo — aislamiento CUDA:** detener o descargar Qwen cuando esté ocioso y repetir sólo el flujo de trabajo normal, sin síntesis de prueba durante un incidente.
6. **Riesgo medio — software de aplicación:** evaluar en una tarea separada una actualización compatible de VS Code y de la extensión Codex; la instalación actual de VS Code 1.103.1 también reporta varias extensiones más nuevas incompatibles.
7. **Riesgo mayor — driver:** evaluar en una tarea separada una actualización del driver NVIDIA, conservando antes un `nvidia-bug-report` y un plan de reversión. No basar esta acción únicamente en el Xid 31 del apagado.

## Comandos exactos para una futura captura

Ejecutar desde la pantalla que siga respondiendo o desde una TTY, **antes de cerrar VS Code, reiniciar GDM o reiniciar el equipo**:

```bash
INCIDENT_TS="$(date +%Y%m%dT%H%M%S%z)"
INCIDENT_DIR="/tmp/vscode-nvidia-freeze-${INCIDENT_TS}"
mkdir -p "$INCIDENT_DIR"

date --iso-8601=seconds > "$INCIDENT_DIR/date.txt"
uptime -s > "$INCIDENT_DIR/uptime-start.txt"
who -b > "$INCIDENT_DIR/last-boot.txt"
journalctl --list-boots --no-pager > "$INCIDENT_DIR/boots.txt"

journalctl -b --since '-20 minutes' -o short-iso --no-pager \
  > "$INCIDENT_DIR/journal-last-20m.log"
journalctl -b -k --since '-20 minutes' -o short-iso --no-pager \
  > "$INCIDENT_DIR/kernel-last-20m.log"
journalctl --user -b --since '-20 minutes' -o short-iso --no-pager \
  > "$INCIDENT_DIR/user-journal-last-20m.log"

nvidia-smi > "$INCIDENT_DIR/nvidia-smi.txt"
nvidia-smi -q > "$INCIDENT_DIR/nvidia-smi-q.txt"
xrandr --listproviders > "$INCIDENT_DIR/xrandr-providers.txt" 2>&1
xrandr --query > "$INCIDENT_DIR/xrandr-query.txt" 2>&1

ps -eo pid,ppid,stat,%cpu,%mem,rss,vsz,etime,cmd --sort=-%cpu \
  > "$INCIDENT_DIR/processes-by-cpu.txt"
ps -eo pid,ppid,stat,%cpu,%mem,rss,vsz,etime,cmd --sort=-%mem \
  > "$INCIDENT_DIR/processes-by-memory.txt"
free -h > "$INCIDENT_DIR/free.txt"
swapon --show > "$INCIDENT_DIR/swap.txt"
df -h > "$INCIDENT_DIR/df-h.txt"
df -i > "$INCIDENT_DIR/df-i.txt"
vmstat 1 10 > "$INCIDENT_DIR/vmstat.txt"

find "$HOME/.config/Code/logs" -maxdepth 4 -type f \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null \
  | sort -r | head -300 > "$INCIDENT_DIR/code-log-files.txt"
find "$HOME/.config/Code/Crashpad" -type f \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null \
  | sort -r | head -100 > "$INCIDENT_DIR/code-crashpad-files.txt"
coredumpctl list --no-pager > "$INCIDENT_DIR/coredumps.txt" 2>&1

tar -C /tmp -czf "${INCIDENT_DIR}.tar.gz" "$(basename "$INCIDENT_DIR")"
printf 'Captura: %s\n' "${INCIDENT_DIR}.tar.gz"
```

Si ya fue necesario reiniciar, capturar inmediatamente el arranque anterior:

```bash
POST_TS="$(date +%Y%m%dT%H%M%S%z)"
POST_DIR="/tmp/vscode-nvidia-post-reboot-${POST_TS}"
mkdir -p "$POST_DIR"
journalctl -b -1 -o short-iso --no-pager > "$POST_DIR/previous-boot.log"
journalctl -b -1 -k -o short-iso --no-pager > "$POST_DIR/previous-kernel.log"
journalctl -b -1 -p warning..alert -o short-iso --no-pager \
  > "$POST_DIR/previous-warnings.log"
last -x -F > "$POST_DIR/last-x.txt"
tar -C /tmp -czf "${POST_DIR}.tar.gz" "$(basename "$POST_DIR")"
printf 'Captura: %s\n' "${POST_DIR}.tar.gz"
```

## Cierre

No se aplicaron correcciones, cambios de código, cambios de configuración, actualizaciones, commits ni push. El repositorio estaba limpio al comenzar; al terminar queda únicamente este reporte como archivo nuevo no versionado.
