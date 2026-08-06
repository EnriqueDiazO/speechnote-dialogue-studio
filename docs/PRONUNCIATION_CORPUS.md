# Corpus de referencia de pronunciación

El corpus fija lecturas matemáticas y científicas que ya fueron revisadas. Su propósito es detectar
regresiones del motor de pronunciación sin convertir la salida actual del algoritmo en verdad por
defecto. Agregar un caso al corpus no crea ni modifica una regla, un diccionario, un proyecto o un
audio.

El diccionario y el corpus cumplen funciones distintas:

- el **diccionario** transforma texto mediante reglas `builtin`, `global`, `project` o `utterance`;
- el **corpus** ejecuta entradas controladas y verifica que el resultado continúe cumpliendo el
  contrato humano aprobado.

## Estructura y carga

Los archivos versionados están bajo `tests/fixtures/pronunciation/`:

```text
manifest.json
approved/{es,en}/*.json
candidates/{es,en}/*.json
deprecated/{es,en}/*.json
```

El cargador descubre todos los `*.json`; no mantiene una lista fija de disciplinas o archivos. Una
categoría nueva se puede añadir al manifiesto y a un archivo sin modificar el motor ni el cargador.
Los archivos son UTF-8, usan Unicode NFC, tienen orden estable y no contienen comentarios, audio,
rutas personales ni datos de proyectos.

`manifest.json` declara `schema_version`, `corpus_version`, idiomas, perfiles predeterminados,
categorías abiertas, conteos por estado y una fecha de validación explícita. Las pruebas no cambian
esa fecha.

## Esquema de un caso

Cada caso contiene:

- identidad: `schema_version` y `case_id` legible, estable y globalmente único;
- selección: `status`, `language`, `profile` y `category`;
- contrato: `written_text`, `expected_spoken_text` y `assertion_mode`;
- comprobaciones: `expected_warning_codes`, `expected_unsupported_fragments`,
  `semantic_anchors`, `forbidden_fragments` y, opcionalmente, `applied_rule_ids`;
- auditoría: `tags`, `notes`, `source_kind`, `source_reference`, `created_at` y `updated_at`.

Las categorías no son una enumeración cerrada. Los perfiles iniciales son `concise`, `classroom`,
`explicit` y `symbolic`, en español (`es`) e inglés (`en`).

### Estados

- `approved`: participa en la regresión y necesita una lectura esperada revisada;
- `candidate`: pasa validación de esquema, pero no fija una salida canónica;
- `deprecated`: se conserva para trazabilidad y no se ejecuta en la suite ordinaria.

### Modos de aserción

- `exact` exige igualdad completa de `spoken_text`;
- `semantic` exige que todas las anclas aparezcan en el orden declarado;
- `warning_only` fija warnings y fragmentos no soportados para sintaxis deliberadamente parcial.

Todos los modos verifican exactamente warnings y fragmentos no soportados. También rechazan cada
`forbidden_fragment`. Los IDs de regla declarados deben aparecer en orden; los candidatos de la UI
sólo incluyen IDs incorporados que sean portables.

## Flujo de revisión

### Crear un candidato

En **Pronunciación → Vista previa**, transforma el texto y abre **Exportar como caso de corpus**.
Desde una intervención activa también está la opción **Exportar lectura como caso candidato**.
Edita `case_id`, categoría, tags y notas, y descarga el JSON.

La descarga contiene el texto escrito original y la lectura efectiva actual. No contiene nombre o
UUID del proyecto, usuario, rutas personales, audio, secretos ni logs, y no escribe en el
repositorio. El aviso de la UI recuerda que esa lectura aún no está aprobada.

Para incorporar el archivo descargado como candidato:

```bash
.venv/bin/python scripts/pronunciation_corpus.py add-candidate --file candidate.json
```

La herramienta preserva el archivo fuente, rechaza IDs duplicados, nunca escribe directamente en
`approved/` y actualiza los conteos de forma determinista.

### Revisar y promover

Lista y examina los casos:

```bash
.venv/bin/python scripts/pronunciation_corpus.py list --status candidate
.venv/bin/python scripts/pronunciation_corpus.py show es-proper-names-haseman-001
```

Revisa manualmente `written_text`, salida actual, lectura esperada, perfil, warnings, fragmentos no
soportados y notas. Corrige explícitamente el JSON candidato si la lectura canónica no coincide con
la salida actual. La herramienta nunca actualiza `expected_spoken_text` por sí sola.

La promoción primero imprime esos datos y no cambia nada sin confirmación:

```bash
.venv/bin/python scripts/pronunciation_corpus.py promote CASE_ID
.venv/bin/python scripts/pronunciation_corpus.py promote CASE_ID --confirm
```

El primer comando sirve como previsualización y termina indicando que falta `--confirm`. El segundo
mueve atómicamente el caso a `approved/<idioma>/<categoría>.json`, actualiza el manifiesto y vuelve
a validar el corpus.

### Deprecar y actualizar una lectura

Un caso obsoleto se conserva con:

```bash
.venv/bin/python scripts/pronunciation_corpus.py deprecate CASE_ID --confirm
```

Para cambiar intencionalmente una lectura canónica, edita el caso aprobado, ejecuta la regresión y
revisa el diff. No hay comando de actualización automática: el cambio de expectativa siempre debe
quedar visible en Git junto con la justificación en `notes` cuando sea útil.

## Validación, estadísticas y reporte

Los accesos habituales son:

```bash
make pronunciation-corpus-validate
make pronunciation-corpus-stats
make pronunciation-corpus-test
```

`validate` revisa esquema, rutas, Unicode, IDs, conflictos, manifiesto y la salida de todos los
aprobados. `stats` informa estados, idiomas, perfiles, categorías, warnings y fragmentos no
soportados. `test` ejecuta todas las pruebas del corpus.

Se puede escribir un reporte reproducible y sin rutas del repositorio en una carpeta temporal
nueva. La ruta no se sobrescribe:

```bash
.venv/bin/python scripts/pronunciation_corpus.py stats \
  --report temporary/pronunciation-corpus-report.json
```

El archivo generado no se versiona. Debe borrarse o conservarse fuera del árbol antes de cerrar el
trabajo.

## Interpretar un fallo

Una regresión muestra `CASE`, `PROFILE`, `WRITTEN`, `EXPECTED`, `ACTUAL`, `FIRST DIFFERENCE` y la
cláusula incumplida. Antes de modificar el esperado, decide cuál de estos hechos ocurrió:

1. el motor introdujo un defecto: corrige el parser o la regla y conserva el caso;
2. la nueva lectura es una mejora deliberada: edita el esperado y documenta la decisión;
3. la expresión es ambigua: pásala a candidato o usa anclas semánticas justificadas;
4. la expresión no es segura de interpretar: usa `warning_only`, conserva el fragmento y fija el
   warning, sin inventar una lectura.

La prueba `test_intentional_expected_output_failure_has_a_readable_diff` altera una expectativa sólo
en memoria, comprueba el mensaje anterior y deja intactos los archivos.

## Política editorial

Las lecturas aprobadas deben ser deterministas, comprensibles y revisadas por significado, no
copias masivas de la salida del motor. Las expresiones maduras prefieren `exact`; `semantic` se
reserva para frases donde la redacción completa puede mejorar sin perder conceptos ni orden.

Nombres propios con pronunciación discutible, entre ellos Haseman, Fredholm, Wiener–Hopf, Mellin y
Calkin, permanecen como candidatos hasta una aprobación explícita. No se promueven por frecuencia
de uso ni por una conjetura del motor.

Las expresiones desconocidas conservan una representación legible, warning y fragmento no
soportado. URL, correo, código, rutas y UUID se prueban como casos de seguridad; el corpus nunca los
convierte en reglas.

## Conteos de la versión 1.0.0

El corpus contiene 113 casos: 108 aprobados, 5 candidatos y 0 deprecados. Hay 83 aprobados en
español y 25 en inglés. Los aprobados usan 102 perfiles `classroom`, 2 `concise`, 2 `explicit` y 2
`symbolic`; sus modos son 103 `exact`, 3 `semantic` y 2 `warning_only`.

Conteos aprobados por categoría:

| Categoría | Casos |
|---|---:|
| algebra | 14 |
| basic_arithmetic | 8 |
| calculus | 21 |
| edge_cases | 11 |
| linear_algebra | 11 |
| machine_learning | 11 |
| mixed_prose_math | 3 |
| operator_theory | 8 |
| probability_statistics | 13 |
| singular_integrals | 8 |

Los cinco candidatos son nombres propios españoles bajo `proper_names`. Las disciplinas cubiertas
incluyen aritmética, álgebra, cálculo, álgebra lineal, probabilidad, estadística, aprendizaje
automático, análisis funcional, teoría de operadores e integrales singulares, además de prosa mixta
y límites de seguridad.

Ejemplo mínimo de candidato:

```json
{
  "schema_version": 1,
  "case_id": "es-calculus-ui-candidate-001",
  "status": "candidate",
  "language": "es",
  "profile": "classroom",
  "category": "calculus",
  "written_text": "$x^2$",
  "expected_spoken_text": "equis al cuadrado",
  "assertion_mode": "exact",
  "source_kind": "ui_export"
}
```

El archivo real puede incluir todos los campos de auditoría. Al importarlo, el esquema completa los
campos opcionales con valores vacíos, pero nunca decide si la lectura merece estado `approved`.
