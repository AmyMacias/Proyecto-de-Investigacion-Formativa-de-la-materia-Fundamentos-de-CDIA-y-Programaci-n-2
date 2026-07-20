"""
Agente LLM del Sistema de Información Inteligente (SII) de horarios.

Este módulo arma el contexto a partir de los resultados del algoritmo
genético (horario optimizado + simulación de imprevisto) y lo inserta en
un prompt de sistema. El agente cumple dos funciones:

  1. Explicar el horario y responder preguntas sobre él.
  2. Sugerir ajustes cuando ocurre un imprevisto (profesor no disponible,
     aula cerrada, etc.).

Si existe una variable de entorno ANTHROPIC_API_KEY, el agente llama a la
API real de Anthropic. Si no, cae en un modo simulado basado en reglas que
usa los mismos datos, útil para la demo y para no depender de una key.
"""

import os
import json


PROMPT_SISTEMA = """Eres el asistente del Sistema de Información Inteligente (SII) \
de planificación de horarios académicos de la carrera. Tienes acceso al horario \
generado por un algoritmo genético (optimizado a partir de restricciones de \
choques de profesores, choques de grupos, disponibilidad horaria, no repetición \
de materias el mismo día y balance de carga por día) y al resultado de una \
simulación de imprevisto (un profesor deja de estar disponible).

Tu rol tiene dos responsabilidades:
1. EXPLICAR el horario y responder preguntas de estudiantes, docentes o \
coordinadores (ej. "¿qué días tiene clase el grupo B?", "¿cuántas clases \
tiene el profesor X?", "¿cuál es el fitness del horario y qué significa?").
2. ANTE UN IMPREVISTO (profesor enfermo, aula cerrada), analizar qué clases \
quedan sin cubrir y proponer una reasignación razonable usando los mismos \
criterios que el algoritmo genético (evitar choques, respetar disponibilidad, \
no sobrecargar un día).

Reglas de estilo:
- Responde en español, de forma breve y concreta.
- Si el dato no está en el contexto que te doy, dilo explícitamente en vez \
de inventarlo.
- Cuando propongas una reasignación, sé específico: materia, grupo, día y \
hora sugeridos, y por qué esa opción no genera conflicto.
- No inventes profesores, materias o grupos que no estén en los datos.
"""


def construir_contexto(resultados: dict, max_clases: int = 100) -> str:
    """Arma el bloque de contexto (horario + antes/después del imprevisto)
    que se inserta en el prompt antes de la pregunta del usuario."""

    horario = resultados["tabla_horario"][:max_clases]
    horario_txt = "\n".join(
        f"- {c['ID Clase']} | {c['Materia']} | Prof. {c['Profesor']} | "
        f"Grupo {c['Grupo']} | {c['Día']} {c['Hora']}"
        for c in horario
    )

    ro = resultados["resultado_optimo"]
    ri = resultados["resultado_imprevisto"]
    afectado = resultados["profesor_afectado"]
    clases_afectadas = resultados["clases_afectadas"]

    clases_afectadas_txt = "\n".join(
        f"- {c['ID Clase']} | {c['Materia']} | Grupo {c['Grupo']} | "
        f"{c['Día']} {c['Hora']} (quedó sin profesor)"
        for c in clases_afectadas
    )

    contexto = f"""
HORARIO OPTIMIZADO (fitness final: {resultados['fitness_final']}/100):
{horario_txt}

INDICADORES DEL HORARIO OPTIMIZADO:
{json.dumps(ro, ensure_ascii=False, indent=2)}

SIMULACIÓN DE IMPREVISTO: el profesor "{afectado}" dejó de estar disponible.
Clases que quedaron sin cubrir:
{clases_afectadas_txt}

INDICADORES DEL HORARIO TRAS EL IMPREVISTO (sin reasignar aún):
{json.dumps(ri, ensure_ascii=False, indent=2)}
"""
    return contexto.strip()


def preguntar_agente(pregunta: str, resultados: dict) -> str:
    """Punto de entrada del agente. Usa la API de Anthropic si hay
    ANTHROPIC_API_KEY configurada; si no, responde en modo simulado."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    contexto = construir_contexto(resultados)

    if api_key:
        return _preguntar_api(pregunta, contexto, api_key)
    return _respuesta_simulada(pregunta, resultados)


def _preguntar_api(pregunta: str, contexto: str, api_key: str) -> str:
    try:
        import anthropic
    except ImportError:
        return (
            "No está instalada la librería 'anthropic' (pip install anthropic). "
            "Mostrando en su lugar la respuesta en modo simulado:\n\n"
            + _respuesta_simulada(pregunta, json.loads(contexto))
        )

    client = anthropic.Anthropic(api_key=api_key)
    mensaje = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=PROMPT_SISTEMA + "\n\nCONTEXTO:\n" + contexto,
        messages=[{"role": "user", "content": pregunta}],
    )
    return "".join(b.text for b in mensaje.content if b.type == "text")


def _respuesta_simulada(pregunta: str, resultados: dict) -> str:
    """Modo simulado (sin llamada a API): usa reglas simples sobre los
    mismos datos para responder preguntas frecuentes. Sirve como demo
    y como referencia de los ejemplos de conversación del informe."""

    p = pregunta.lower()
    horario = resultados["tabla_horario"]

    # Pregunta por un grupo
    for c in horario:
        grupo = c["Grupo"].lower()
        if f"grupo {grupo}" in p or (len(p.split()) < 6 and grupo == p.strip()[-1:]):
            clases_grupo = [x for x in horario if x["Grupo"] == c["Grupo"]]
            dias = sorted(set(x["Día"] for x in clases_grupo))
            return (
                f"El grupo {c['Grupo']} tiene {len(clases_grupo)} clases, "
                f"distribuidas en: {', '.join(dias)}."
            )

    # Pregunta sobre el imprevisto (se evalúa antes que la búsqueda por
    # nombre de profesor, porque el profesor afectado también aparece
    # mencionado en preguntas sobre el imprevisto)
    palabras_imprevisto = ("imprevisto", "enferm", "no disponible", "aula cerrada", "ausente", "falta")
    if any(w in p for w in palabras_imprevisto) or resultados["profesor_afectado"].lower() in p:
        afectado = resultados["profesor_afectado"]
        clases = resultados["clases_afectadas"]
        detalle = "; ".join(
            f"{c['Materia']} (grupo {c['Grupo']}, {c['Día']} {c['Hora']})" for c in clases
        )
        return (
            f"Ante la ausencia de {afectado}, quedaron sin cubrir {len(clases)} clases: "
            f"{detalle}. Sugerencia: reasignar cada una a un docente disponible en ese "
            f"mismo bloque horario que no tenga ya una clase asignada con ese grupo, "
            f"priorizando mantener el balance de carga entre días."
        )

    # Pregunta por un profesor específico
    for profesor in resultados["carga_docente"]:
        if profesor.lower() in p:
            n = resultados["carga_docente"][profesor]
            return f"El profesor {profesor} tiene {n} clases asignadas en el horario optimizado."

    # Pregunta por el fitness / calidad
    if "fitness" in p or "calidad" in p or "puntaje" in p:
        ro = resultados["resultado_optimo"]
        base = (
            f"El horario optimizado tiene un fitness de {resultados['fitness_final']}/100 "
            f"(partió de {resultados['fitness_inicial']}/100 antes de optimizar). "
        )
        if resultados["fitness_final"] >= 100:
            return base + (
                "Llegó al máximo: no hay choques de profesor ni de grupo, no hay materias repetidas "
                "el mismo día para ningún grupo, no hay violaciones de disponibilidad docente, y la "
                "carga está perfectamente balanceada entre los días."
            )
        return base + (
            f"No llega a 100 porque aún quedan {ro['materias_repetidas']} casos de materia "
            f"repetida el mismo día para un grupo y una diferencia de "
            f"{ro['distribucion_desigual']} clases entre el día con más carga y el de menos."
        )

    return (
        "Puedo responder preguntas sobre el horario (por grupo, por profesor), "
        "sobre el fitness/calidad del horario, o sobre la simulación del imprevisto. "
        "¿Podrías reformular tu pregunta mencionando el grupo, profesor o tema puntual?"
    )


# Ejemplos de conversación para el informe (sección "Integración del agente LLM")
EJEMPLOS_CONVERSACION = [
    {
        "pregunta": "¿Qué días tiene clase el grupo B?",
        "respuesta_tipo": "explicación del horario",
    },
    {
        "pregunta": "¿Cuál es el fitness del horario y qué significa?",
        "respuesta_tipo": "explicación del horario / métricas",
    },
    {
        "pregunta": "Elena Rostova se enfermó, ¿qué clases quedan sin cubrir y qué propones?",
        "respuesta_tipo": "reacción ante imprevisto",
    },
]
