import os
from datetime import datetime
import streamlit as st
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv
from agente import Agent, SANDBOX_DIR

load_dotenv()


def saludo_segun_hora():
    hora = datetime.now().hour
    if hora < 12:
        return "Buenos días"
    elif hora < 19:
        return "Buenas tardes"
    else:
        return "Buenas noches"

MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
MAX_TOOL_ITERATIONS = 10

# Tipos de archivo que se pueden adjuntar desde la barra de mensaje
ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg", "webp", "gif", "txt", "md", "py", "csv", "json", "pdf"]

st.set_page_config(page_title="Mi primer agente de IA", page_icon="✨", layout="centered")

# ---------- Estilos para acercarse a la estética de Claude ----------
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background-color: #262624;
    }
    .block-container {
        max-width: 780px;
        padding-top: 3rem;
    }
    .stChatMessage {
        background-color: transparent;
    }
    div[data-testid="stChatInput"] {
        border-radius: 18px;
    }
    .greeting-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 2.6rem;
        font-weight: 400;
        text-align: center;
        color: #E8E6E1;
        margin-top: 4rem;
        margin-bottom: 2rem;
    }
    div[data-testid="column"] button {
        border-radius: 999px !important;
        border: 1px solid #4a4a47 !important;
        background-color: #33322f !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Validar configuración ----------
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("❌ Falta OPENROUTER_API_KEY en tu archivo .env. Revisa la configuración antes de continuar.")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# ---------- Estado de la sesión ----------
if "agent" not in st.session_state:
    st.session_state.agent = Agent(web_mode=True)

agent = st.session_state.agent

# ---------- Barra lateral ----------
with st.sidebar:
    st.markdown("### ✨ Mi agente")
    if st.button("➕ Nueva conversación", use_container_width=True):
        agent.reset()
        st.rerun()
    st.markdown("---")
    st.caption("El agente puede leer y crear archivos en `workspace/`, leer archivos externos (con tu permiso) y analizar imágenes. Adjuntá archivos con el 📎 de la barra de mensaje.")

# ---------- ¿Hay conversación empezada? ----------
visible_messages = [
    m for m in agent.messages
    if m["role"] in ("user", "assistant") and (m["role"] == "user" or m.get("content"))
]

def send_message(text):
    """Envía un mensaje como si lo hubiera escrito el usuario y corre el loop del agente."""
    with st.chat_message("user"):
        st.markdown(text)
    agent.messages.append({"role": "user", "content": text})
    run_agent_loop()
    st.rerun()


def call_model_streaming(placeholder):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=agent.messages,
        tools=agent.tools,
        stream=True,
    )
    content = ""
    tool_calls = {}
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
            placeholder.markdown(content + "▌")
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                if tc_delta.id:
                    tool_calls[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls[idx]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls[idx]["arguments"] += tc_delta.function.arguments
    placeholder.markdown(content)
    return content, tool_calls


def run_agent_loop():
    """Llama al modelo repetidamente hasta que ya no pida más herramientas,
    o hasta que quede una acción pendiente de confirmación."""
    iterations = 0
    while True:
        iterations += 1
        if iterations > MAX_TOOL_ITERATIONS:
            st.warning("⚠️ Se alcanzó el límite de llamadas a herramientas para esta respuesta.")
            break

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                content, tool_calls = call_model_streaming(placeholder)
            except RateLimitError:
                st.warning("⚠️ Se alcanzó el límite de solicitudes. Intenta de nuevo en unos segundos.")
                break
            except APIConnectionError:
                st.warning("⚠️ No se pudo conectar con OpenRouter. Revisa tu conexión a internet.")
                break
            except APIError as e:
                st.warning(f"⚠️ Error de la API: {e}")
                break
            except Exception as e:
                st.warning(f"⚠️ Ocurrió un error inesperado: {e}")
                break

        status = agent.process_message_web(content, tool_calls)

        if status == "pending":
            break
        if status == "done":
            agent.save_history()
            break


# ---------- Pantalla de bienvenida (sin mensajes todavía) ----------
if not visible_messages and not agent.pending_tool_call:
    st.markdown(f"<div class='greeting-title'>✻ {saludo_segun_hora()}</div>", unsafe_allow_html=True)

    quick_starts = [
        ("📂 Listar archivos", "Listame los archivos que hay en workspace."),
        ("📝 Crear un archivo", "Creá un archivo de notas.txt en workspace con un saludo dentro."),
        ("🖼️ Analizar una imagen", "Quiero que analices una imagen. Te la voy a adjuntar."),
        ("📖 Leer un archivo externo", "Necesito que leas un archivo que está fuera de workspace."),
    ]
    cols = st.columns(2)
    for i, (label, message) in enumerate(quick_starts):
        with cols[i % 2]:
            if st.button(label, use_container_width=True, key=f"quick_{i}"):
                send_message(message)

# ---------- Mostrar historial de la conversación ----------
else:
    for msg in visible_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------- Si hay una acción pendiente, mostrar botones de confirmación ----------
if agent.pending_tool_call:
    tc = agent.pending_tool_call
    path = tc["args"].get("path", "?")
    if tc["name"] == "edit_file":
        st.warning(f"⚠️ El agente quiere modificar/crear el archivo **{path}** en `workspace/`. ¿Lo permites?")
    else:
        st.warning(f"⚠️ El agente quiere **leer un archivo fuera de workspace**: **{path}**. ¿Lo permites?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Permitir", use_container_width=True):
            agent.resolve_pending_edit(approved=True)
            run_agent_loop()
            st.rerun()
    with col2:
        if st.button("❌ Rechazar", use_container_width=True):
            agent.resolve_pending_edit(approved=False)
            run_agent_loop()
            st.rerun()

# ---------- Barra de mensaje, con adjuntar archivo integrado (📎) ----------
else:
    prompt = st.chat_input(
        "¿Cómo puedo ayudarte hoy?",
        accept_file="multiple",
        file_type=ALLOWED_FILE_TYPES,
    )

    if prompt:
        text = (prompt.text or "").strip()
        attached_paths = []

        for uploaded_file in prompt.files:
            uploads_dir = os.path.join(SANDBOX_DIR, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            saved_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            attached_paths.append(f"uploads/{uploaded_file.name}")

        if attached_paths:
            archivos_txt = ", ".join(attached_paths)
            if text:
                full_message = f"{text}\n\n(Archivos adjuntos: {archivos_txt})"
            else:
                full_message = f"Adjunté estos archivos: {archivos_txt}. Analizalos o leelos según corresponda y contame qué encontraste."
        else:
            full_message = text

        if full_message:
            send_message(full_message)