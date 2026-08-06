# Seguridad de Qwen cuando NVIDIA también maneja pantallas

## Límite reconocido

En el equipo auditado, la RTX 3060 Ti de 8 GiB ejecuta CUDA y también maneja dos pantallas dentro
de una sesión X11 híbrida Intel/NVIDIA. Xorg, GNOME/Mutter, Electron y Qwen comparten driver,
VRAM y recursos de presentación. El modo seguro de Dialogue Studio reduce y aísla el riesgo; no
puede garantizar que una pantalla nunca se congele ni recuperar un driver NVIDIA que ya quedó
bloqueado.

La causa más probable del incidente auditado fue la interacción del proceso GPU de
VS Code/Electron con GNOME/Mutter en esa sesión híbrida, con confianza media. La relación directa
con Qwen fue baja en ese incidente concreto, aunque las síntesis Qwen han coincidido repetidamente
con congelamientos observados. Por eso la política no presupone que Qwen sea la única causa.

## Flujo recomendado

1. Abre el proyecto desde una terminal externa con `code --disable-gpu .`, o cierra VS Code antes
   de sintetizar.
2. Cierra Firefox, Electron y otras aplicaciones gráficas pesadas que no sean necesarias.
3. Abre **Seguridad GPU de Qwen** y ejecuta el preflight.
4. No continúes si aparece **Generación bloqueada para proteger la sesión gráfica**.
5. Confirma una sola vez la advertencia de GPU compartida para esa sesión de la aplicación.
6. Genera secuencialmente; la galería y los lotes nunca ejecutan dos inferencias a la vez.
7. Descarga el modelo o detén el worker al terminar. El worker descarga el modelo a los 120 s y
   termina a los 300 s de inactividad con la política predeterminada.

Los umbrales predeterminados fueron elegidos prudentemente para este equipo, no son universales.
La política completa se puede revisar y editar en la UI y se guarda como
`config/qwen-gpu-safety-policy.json` dentro de la raíz de datos de la aplicación.

## Qué comprueba el preflight

El controller consulta sin privilegios:

- nombre, driver, temperatura, utilización y VRAM mediante `nvidia-smi`;
- procesos de cómputo y procesos gráficos NVIDIA;
- CUDA y BF16 mediante un proceso corto del runtime Qwen externo;
- tipo de sesión y `DISPLAY`;
- eventos recientes filtrados de `journalctl -k`, incluidos NVRM, Xid, MMU, modeset, motor de
  pantalla, hang y timeout;
- workers Qwen no reconocidos y síntesis en curso.

La política predeterminada es fail-closed. Falta de datos esenciales, CUDA/BF16 ausentes, Xid
reciente, temperatura o VRAM fuera de umbral, otro worker, síntesis activa o estado inconsistente
bloquean la carga. Xorg y GNOME no bloquean por sí solos; son procesos esperados. VS Code,
Electron, Firefox y Speech Note acelerados producen advertencias y nunca se cierran
automáticamente.

## Aislamiento y recuperación

La arquitectura es:

```text
Dialogue Studio → HTTP local → controller Qwen → worker descartable → CUDA o CPU explícita
```

El controller no importa Torch ni Qwen. Crea un único worker con `subprocess`, nunca con `fork`
después de CUDA. El worker usa exclusivamente el modelo 0.6B autorizado, `cuda:0`, BF16, SDPA e
`inference_mode` en el modo normal. Existen límites separados para arranque, carga y síntesis.

Durante carga y generación, el controller sondea el worker y NVIDIA cada 1.5 s; consulta el
journal cada 10 s. Un Xid nuevo enclava `gpu_fault`, detiene sólo el worker administrado y bloquea
trabajos posteriores. No intenta recargar automáticamente. Temperatura o VRAM críticas también
detienen la cola. `SIGKILL` sólo se usa contra ese worker si no responde a `SIGTERM` dentro del
periodo de gracia.

El audio se valida para descartar NaN/infinitos y se escribe con nombre `.partial`; únicamente un
RIFF/WAVE válido se publica con `os.replace`. Una toma final previa nunca se borra por un fallo.
Terminar el worker libera toda su VRAM, pero no se afirma que eso recupere un driver ya bloqueado.

## Diagnóstico antes de reiniciar

Usa **Descargar diagnóstico** antes de reiniciar la sesión o el sistema. El JSON contiene política,
preflight, métricas, versiones, estado y código de salida del worker, timeout, errores filtrados,
eventos Xid/NVRM y hashes/rutas relativas de salidas. No incluye texto del guion, audio, tokens,
rutas personales completas ni variables de entorno.

Si aparece un Xid o una pantalla deja de responder, no continúes con otra voz ni permitas un
reinicio automático del worker. Conserva el diagnóstico y considera el resultado:

```text
APPLICATION_LEVEL_MITIGATION_INSUFFICIENT
```

## CPU de emergencia

El modo CPU usa el mismo modelo 0.6B y el mismo entorno, en un worker separado con `float32` y
SDPA. Está desactivado por defecto, exige `allow_cpu_fallback=true` y una confirmación explícita en
cada sesión, y nunca se activa como consecuencia automática de un fallo GPU.

La evaluación controlada del 5 de agosto de 2026, con `Prueba breve.`, midió 2.51 s de carga,
4.04 s de generación, 6.55 s dentro del proceso, audio finito a 24 kHz y 5,887 MiB
(aproximadamente 5.75 GiB) de RAM máxima, sin swap. Es viable en este equipo, pero se presenta como:
**Muy lento. Sólo para diagnóstico o frases cortas.** Estas cifras no son una promesa para otros
textos o equipos.

## Solución robusta fuera de esta tarea

La mitigación más robusta es separar la GPU de presentación de la GPU de cómputo: usar Intel para
todas las pantallas y reservar NVIDIA para CUDA, ya sea físicamente o mediante una configuración
cuidadosamente planificada. Cualquier cambio de conexiones, PRIME, Xorg, driver, kernel, GDM o
GNOME debe realizarse en una tarea separada, con un plan de recuperación. Dialogue Studio y esta
tarea no aplican ninguno de esos cambios automáticamente.
