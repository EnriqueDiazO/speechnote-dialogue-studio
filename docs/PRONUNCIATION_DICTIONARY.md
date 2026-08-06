# Diccionario de pronunciación

SpeechNote Dialogue Studio conserva dos representaciones distintas de cada intervención:

- `written_text` es exactamente lo escrito por el usuario. La UI, los subtítulos, el guion y
  `text` en proyectos antiguos siguen usando esta representación.
- `spoken_text` es una representación derivada y reproducible. Sólo esta representación se envía
  al proveedor TTS.

La secuencia común es:

```text
written_text
  → segmentación
  → reglas y glosarios
  → lector matemático
  → números, unidades y siglas
  → spoken_text
  → Speech Note o Qwen3-TTS
  → audio
```

Los proveedores no contienen reglas de pronunciación. Para la misma intervención, reglas y
perfil, Speech Note y Qwen reciben el mismo `spoken_text`.

## Arquitectura

La implementación vive en `dialogue_studio/pronunciation/`:

- `models.py`: reglas, perfiles, resultados, warnings y límites de validación;
- `engine.py`: precedencia, selección determinista, traza y hashes;
- `segmentation.py`: prosa, matemáticas, código, URL, correo y ruta;
- `glossary.py` y `resources/{es,en}/*.json`: recursos incorporados versionados;
- `math_speech.py`: parser limitado y recursivo de LaTeX/Unicode;
- `linguistics.py`: números, fechas, ordinales, unidades y siglas;
- `import_export.py`: JSON/CSV, almacenamiento global, conflictos y términos pendientes;
- `ui.py`: vista previa, editores y controles compactos por intervención.

`dialogue_studio.service.effective_pronunciation_result` es la entrada común del dominio. La
síntesis calcula ese resultado antes de entrar en estado `generating`.

## Perfiles

`PronunciationProfile` contiene:

- `enabled` y `language` (`es` o `en` inicialmente);
- `math_style`: `concise`, `classroom`, `explicit` o `symbolic`;
- `acronym_policy`: `custom`, `spell_out`, `read_as_word` o `preserve`;
- `number_style`: `natural`, `digits` o `preserve`;
- `unit_style`: `natural`, `spell_out` o `preserve`;
- `punctuation_style`: `natural`, `explicit` o `preserve`.

Los perfiles matemáticos se interpretan así:

- `concise`: lectura corta, por ejemplo «a sobre b»;
- `classroom`: lectura docente, por ejemplo «a dividido entre b»;
- `explicit`: anuncia numerador, denominador y agrupaciones;
- `symbolic`: lectura deliberadamente literal para inspección.

La puntuación escrita no se elimina ni se sustituye en `written_text`. El texto hablado conserva
pausas ordinarias sin introducir SSML, porque no todos los proveedores lo admiten.

## Reglas y precedencia

Una regla tiene UUID, alcance, idioma, tipo, patrón, reemplazo, estado, prioridad, sensibilidad a
mayúsculas, coincidencia de palabra completa, categoría abierta, notas, fechas, uso e historial.
Los tipos iniciales son `literal`, `phrase`, `acronym`, `regex` y `math_alias`.

La precedencia, de mayor a menor, es:

1. override manual de la intervención;
2. reglas de la intervención;
3. reglas del proyecto;
4. reglas globales;
5. reglas incorporadas.

Dentro de un alcance se ordena por prioridad descendente, coincidencia más larga y `rule_id` como
desempate estable. La transformación no vuelve a procesar sus propios reemplazos: no es recursiva.
Las reglas desactivadas se ignoran y los ciclos literales simples se omiten con warning.

`whole_word` evita casos como leer `pi` dentro de `pipeline`. Las regex están en opciones
avanzadas, tienen longitud limitada, se compilan antes de guardarse y rechazan cuantificadores
anidados obvios. No se usa `eval`, no se ejecuta LaTeX y no se invocan comandos desde las reglas.

## Diccionarios

### Incorporado

Los recursos incorporados son JSON con `schema_version: 1`. Incluyen español e inglés y cubren:

- letras griegas y variantes;
- operadores, relaciones, conjuntos y cálculo;
- álgebra lineal, probabilidad y estadística;
- vocabulario de redes neuronales y aprendizaje automático;
- siglas iniciales como MSE, MAE, SGD, CNN, RNN, LSTM y FDR;
- unidades Hz, kHz, MHz, GB, MB, ms, s, min, h, m, cm, mm, kg, g y °C.

Las reglas incorporadas son visibles y buscables, pero no se editan. Una regla global o de
proyecto puede sobrescribirlas. Para extender el catálogo basta añadir otro recurso validado bajo
`resources/es/` o `resources/en/`; el motor no contiene una lista fija de archivos o categorías.

### Global

El diccionario global se guarda fuera del repositorio mediante `AppPaths`:

```text
<Música>/SpeechNote Dialogue Studio/config/pronunciation-dictionary.json
```

Su formato es:

```json
{
  "schema_version": 1,
  "rules": []
}
```

La escritura es atómica. Si el JSON está corrupto, la aplicación conserva el original, crea una
copia `.corrupt-<fecha>.bak`, abre un diccionario vacío recuperable y muestra un warning.

### Proyecto e intervención

`project.json` guarda únicamente las reglas creadas por el usuario, nunca una copia del catálogo
incorporado. Cada intervención puede guardar:

- `use_pronunciation_engine`;
- `manual_spoken_text_override`;
- `utterance_rules`;
- último `spoken_text` y metadata de síntesis.

Un override manual nunca cambia `text` ni `written_text`. Las reglas locales se pueden promover al
proyecto o al diccionario global desde la UI.

## Segmentación protegida

El motor reconoce prosa, matemáticas inline/display, código entre backticks, bloques de código,
URL, correo y rutas. Reconoce `$...$`, `$$...$$`, `\(...\)`, `\[...\]` y fórmulas Unicode
evidentes.

Código, URL, correo y rutas no se alteran automáticamente. Sólo una regla explícita con alcance
`utterance` puede cambiarlos. Los segmentos conservan posiciones originales para la traza.

## Ecuaciones soportadas

El lector es intencionalmente limitado, recursivo y determinista. Soporta:

- grupos `{}`, `()` y `[]`;
- subíndices y superíndices, incluidos cuadrado, cubo, inversa y transpuesta;
- `\frac`, `\sqrt` y raíces con índice;
- `\sum`, `\prod`, `\int`, `\iint`, `\iiint` y `\oint`, con límites;
- `\lim`, límites laterales y flechas;
- `\sin`, `\cos`, `\tan`, `\log`, `\ln`, `\exp`, `\max`, `\min`, `\arg\max` y
  `\arg\min`;
- `\mathbb{R}`, `\mathbb{C}`, `\mathbb{N}` y `\mathbb{Z}`;
- `\vec`, `\mathbf`, `\hat`, normas, valor absoluto y producto interno;
- matrices básicas `matrix`, `pmatrix`, `bmatrix` y `vmatrix`;
- símbolos Unicode equivalentes cubiertos por el glosario.

Las pruebas bilingües revisan semánticamente, entre otras:

```text
θ_{t+1}=θ_t-η∇L(θ_t)
y = \sigma(Wx+b)
\frac{\partial L}{\partial w_{ij}}
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
\lim_{n\to\infty}\frac{1}{n}\sum_{i=1}^{n}X_i
\int_a^b f(x)\,dx
```

No se intenta implementar LaTeX completo. Un comando desconocido como
`\unknowncommand{x}` se conserva de forma legible, se registra como fragmento no soportado y
produce un warning; no genera traceback ni bloquea la edición. El usuario puede corregirlo con un
override. Para sintetizar tras un fallo excepcional del motor existe un fallback al texto escrito,
pero debe marcarse explícitamente en la tarjeta.

## Números, unidades y siglas

La lectura numérica cubre enteros, decimales, porcentajes, rangos, notación científica, fechas y
ordinales básicos. Los límites de palabra evitan modificar UUID y hashes. Las unidades se ordenan
por longitud para distinguir, por ejemplo, `m`, `mm` y `MHz`, y ajustan singular/plural cuando es
seguro.

Las pronunciaciones personalizadas siempre tienen prioridad sobre la política de siglas. Con
`spell_out`, una sigla desconocida se deletrea según el idioma y también aparece como término por
revisar. Con `read_as_word` se intenta leer como palabra; `preserve` la conserva.

Nombres propios y pronunciaciones discutibles no forman parte del catálogo como verdades
universales. Qwen, Haseman, Fredholm, MOFA2, DIABLO, Praat o Parselmouth se agregan desde la UI si
el usuario confirma su lectura.

## Vista previa, traza y términos por revisar

La sección **Pronunciación** ofrece vista previa independiente del diálogo, texto hablado, reglas
aplicadas, fragmentos no soportados y warnings. La prueba de voz usa el proveedor y la voz elegidos
sin cambiar la intervención.

Cada aplicación de regla registra `rule_id`, tipo, alcance, posición, fragmento original,
reemplazo y prioridad. Las tarjetas sólo muestran un resumen; la traza completa se abre bajo
demanda.

La bandeja **Términos por revisar** es una lista de candidatos, no un diccionario automático. El
usuario puede crear una regla, ignorar una aparición, ignorar siempre o posponer. Se guarda contexto,
origen y frecuencia. Esto evita convertir errores tipográficos o identificadores accidentales en
reglas permanentes.

## Importación y exportación

Se admiten JSON y CSV. Antes de importar se muestran reglas válidas, rechazadas, duplicados,
sombras, ciclos y referencias regex inválidas. El usuario selecciona individualmente qué reglas
aplicar y elige uno de estos modos incrementales:

- agregar sólo nuevas;
- actualizar por `rule_id`;
- importar desactivadas para revisar.

No existe una operación silenciosa que reemplace el diccionario completo. Los conflictos del
editor se pueden descartar, fusionar con una regla existente o guardar sólo mediante confirmación
explícita.

El archivo demostrativo `examples/pronunciation_dictionary_es.json` es importable y editable. Sus
lecturas son ejemplos, no afirmaciones universales.

## Fingerprint y audio stale

La huella de audio incluye texto escrito, texto hablado efectivo, idioma, perfil, `rules_hash`,
override, proveedor, modelo, voz, instrucción y parámetros TTS. La metadata de una síntesis guarda:

- `written_text_hash` y `spoken_text_hash`;
- `pronunciation_rules_hash`;
- `pronunciation_engine_version`;
- reglas aplicadas y warnings.

Al editar texto, perfil, override o una regla aplicable, el audio pasa a `stale`; el WAV anterior no
se borra ni se regenera automáticamente. Los cambios de reglas se comparan por efecto para no
marcar intervenciones no afectadas. El ZIP portable conserva `written_text`, `spoken_text`, perfil,
hashes, reglas aplicadas y warnings; los guiones siguen usando el texto escrito.

## Compatibilidad y solución de problemas

Un proyecto antiguo sin campos de pronunciación se abre en memoria con motor activo, perfil según
idioma, sin reglas ni overrides. Abrirlo no escribe ni migra el archivo; sólo un guardado explícito
serializa los defaults.

- Si una palabra no cambia, revisa alcance, idioma, estado, `whole_word` y prioridad.
- Si aparece un warning de ciclo, desactiva una de las reglas opuestas.
- Si una regex se rechaza, corrige el patrón o las referencias de grupos.
- Si una ecuación queda parcial, usa la traza y crea un alias o un override local.
- Si el diccionario global está corrupto, conserva el archivo y su copia de recuperación antes de
  reconstruir o importar reglas.
- Si el audio figura `stale`, la toma anterior sigue disponible; pulsa **Regenerar** cuando quieras.
- Si el preprocesamiento falla, corrige la regla. Usa el fallback explícito sólo si realmente deseas
  que el proveedor reciba el texto escrito.
