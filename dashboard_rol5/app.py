import json
import pandas as pd
import streamlit as st
import plotly.express as px

from agente_llm import preguntar_agente

st.set_page_config(
    page_title="SII - Horarios Académicos",
    page_icon="🗓️",
    layout="wide",
)


@st.cache_data
def cargar_resultados():
    with open("resultados_horario.json", "r", encoding="utf-8") as f:
        return json.load(f)


resultados = cargar_resultados()
tabla_horario = pd.DataFrame(resultados["tabla_horario"])

st.title("🗓️ Sistema de Información Inteligente — Horarios Académicos")
st.caption(
    "Horario generado por algoritmo genético · Dashboard e integración del agente LLM (Rol 5)"
)

tab_horario, tab_visual, tab_imprevisto, tab_agente = st.tabs(
    ["📋 Horario e indicadores", "📊 Visualizaciones", "⚠️ Simulación de imprevisto", "🤖 Asistente"]
)

# ---------------- TAB 1: Horario e indicadores ----------------
with tab_horario:
    col1, col2, col3, col4 = st.columns(4)
    ro = resultados["resultado_optimo"]
    col1.metric("Fitness final", f"{resultados['fitness_final']}/100",
                delta=f"{resultados['fitness_final'] - resultados['fitness_inicial']} vs. inicial")
    col2.metric("Choques profesor/grupo", ro["conflictos_profesor"] + ro["conflictos_grupo"])
    col3.metric("Materias repetidas mismo día", ro["materias_repetidas"])
    col4.metric("Desbalance de carga (días)", ro["distribucion_desigual"])

    st.subheader("Horario optimizado")
    grupos = ["Todos"] + sorted(tabla_horario["Grupo"].unique().tolist())
    profesores = ["Todos"] + sorted(tabla_horario["Profesor"].unique().tolist())
    c1, c2 = st.columns(2)
    grupo_sel = c1.selectbox("Filtrar por grupo", grupos)
    prof_sel = c2.selectbox("Filtrar por profesor", profesores)

    tabla_filtrada = tabla_horario.copy()
    if grupo_sel != "Todos":
        tabla_filtrada = tabla_filtrada[tabla_filtrada["Grupo"] == grupo_sel]
    if prof_sel != "Todos":
        tabla_filtrada = tabla_filtrada[tabla_filtrada["Profesor"] == prof_sel]

    st.dataframe(tabla_filtrada, use_container_width=True, hide_index=True)

# ---------------- TAB 2: Visualizaciones ----------------
with tab_visual:
    st.subheader("Distribución de clases por día y hora")
    orden_dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    heatmap_data = (
        tabla_horario.groupby(["Día", "Hora"]).size().reset_index(name="Clases")
    )
    fig_heatmap = px.density_heatmap(
        heatmap_data, x="Hora", y="Día", z="Clases",
        category_orders={"Día": orden_dias},
        color_continuous_scale="Blues",
        text_auto=True,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Convergencia del fitness")
        hist_df = pd.DataFrame({
            "Generación": range(1, len(resultados["historial_fitness"]) + 1),
            "Fitness": resultados["historial_fitness"],
        })
        st.plotly_chart(
            px.line(hist_df, x="Generación", y="Fitness", markers=True),
            use_container_width=True,
        )

    with col_b:
        st.subheader("Fitness: inicial vs. optimizado")
        comp_df = pd.DataFrame({
            "Horario": ["Inicial", "Optimizado"],
            "Fitness": [resultados["fitness_inicial"], resultados["fitness_final"]],
        })
        st.plotly_chart(
            px.bar(comp_df, x="Horario", y="Fitness", color="Horario",
                   color_discrete_sequence=["#FF7F7F", "#66C2A5"]),
            use_container_width=True,
        )

    st.subheader("Carga horaria por docente")
    carga_df = pd.DataFrame(
        list(resultados["carga_docente"].items()), columns=["Profesor", "Clases"]
    ).sort_values("Clases", ascending=False)
    st.plotly_chart(
        px.bar(carga_df, x="Profesor", y="Clases", color_discrete_sequence=["#66B3FF"]),
        use_container_width=True,
    )

# ---------------- TAB 3: Simulación de imprevisto ----------------
with tab_imprevisto:
    afectado = resultados["profesor_afectado"]
    st.subheader(f"Imprevisto simulado: {afectado} no disponible")

    ri = resultados["resultado_imprevisto"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Antes (horario optimizado)**")
        st.json(resultados["resultado_optimo"])
    with col2:
        st.markdown("**Después (sin el profesor afectado, sin reasignar)**")
        st.json(ri)

    st.markdown("**Clases que quedaron sin cubrir:**")
    st.dataframe(pd.DataFrame(resultados["clases_afectadas"]), use_container_width=True, hide_index=True)

    fitness_antes = resultados["resultado_optimo"]["fitness"]
    fitness_despues = ri["fitness"]
    n_afectadas = len(resultados["clases_afectadas"])

    if fitness_despues > fitness_antes:
        explicacion = (
            f"El fitness subió ({fitness_antes} → {fitness_despues}) porque, al quitar clases del "
            "horario, también se quitan penalizaciones asociadas a ellas (por ejemplo, materia "
            "repetida o desbalance de carga). Esto no significa que el horario haya mejorado: en "
            f"realidad quedaron {n_afectadas} clases sin profesor asignado."
        )
    elif fitness_despues < fitness_antes:
        explicacion = (
            f"El fitness bajó ({fitness_antes} → {fitness_despues}) porque, al remover las clases del "
            f"profesor afectado, se generaron nuevas penalizaciones (por ejemplo, desbalance de carga "
            "entre días al quedar un día con menos clases que otros). Además, siguen quedando "
            f"{n_afectadas} clases sin profesor asignado, que es el problema real a resolver."
        )
    else:
        explicacion = (
            f"El fitness se mantuvo igual ({fitness_antes} → {fitness_despues}), pero eso no significa "
            f"que el horario esté bien: quedaron {n_afectadas} clases sin profesor asignado."
        )

    st.info(
        explicacion + " Por eso el agente LLM (pestaña Asistente) es útil aquí: interpreta el resultado "
        "numérico y propone una reasignación concreta."
    )

# ---------------- TAB 4: Agente LLM ----------------
with tab_agente:
    st.subheader("Asistente del SII")
    st.caption(
        "Responde preguntas sobre el horario y sugiere reasignaciones ante el imprevisto simulado. "
        "Si no hay ANTHROPIC_API_KEY configurada, funciona en modo simulado con reglas sobre los mismos datos."
    )

    ejemplos = [
        "¿Qué días tiene clase el grupo B?",
        "¿Cuál es el fitness del horario y qué significa?",
        "Elena Rostova se enfermó, ¿qué clases quedan sin cubrir y qué propones?",
    ]
    st.markdown("**Prueba con:** " + " · ".join(f"`{e}`" for e in ejemplos))

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for rol, msg in st.session_state.chat:
        with st.chat_message(rol):
            st.markdown(msg)

    pregunta = st.chat_input("Escribe tu pregunta sobre el horario...")
    if pregunta:
        st.session_state.chat.append(("user", pregunta))
        with st.chat_message("user"):
            st.markdown(pregunta)

        respuesta = preguntar_agente(pregunta, resultados)
        st.session_state.chat.append(("assistant", respuesta))
        with st.chat_message("assistant"):
            st.markdown(respuesta)
