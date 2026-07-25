import os
from datetime import datetime
import streamlit as st
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv
from agente import Agent, SANDBOX_DIR, list_conversations

load_dotenv()


def saludo_segun_hora():
    hora = datetime.now().hour
    if hora < 12:
        return "Buenos días"
    elif hora < 19:
        return "Buenas tardes"
    else:
        return "Buenas noches"

# El modelo de chat se define más abajo, según si elegís nube (OpenRouter) o local (Ollama)
MAX_TOOL_ITERATIONS = 10

# Tipos de archivo que se pueden adjuntar desde la barra de mensaje
ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg", "webp", "gif", "txt", "md", "py", "csv", "json", "pdf"]

st.set_page_config(page_title="Mi primer agente de IA", page_icon="✨", layout="centered")

# ---------- Estilos para acercarse a la estética de Claude ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap');

    #MainMenu, footer, header {visibility: hidden;}

    :root {
        --bg: #14171A;
        --surface: #1D2124;
        --border: #2C3236;
        --accent: #4FB6A6;
        --accent-dim: #33453F;
        --warn: #E8A33D;
        --text: #E7E9EA;
        --text-muted: #838B90;
    }

    .stApp { background-color: var(--bg); }
    .block-container { max-width: 760px; padding-top: 2.5rem; }

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--text); }
    code, .stCode, div[data-testid="stChatInput"] textarea { font-family: 'IBM Plex Mono', monospace !important; }

    /* Eyebrow tipo bitácora, encima del saludo */
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        color: var(--accent);
        text-align: center;
        margin-bottom: 0.4rem;
    }
    .greeting-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.1rem;
        font-weight: 500;
        text-align: center;
        color: var(--text);
        margin-top: 0.2rem;
        margin-bottom: 2.2rem;
    }

    /* Mensajes de chat con borde izquierdo tipo "entrada de registro" */
    .stChatMessage {
        background-color: var(--surface) !important;
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 6px;
        padding: 0.6rem 0.9rem !important;
    }
    .stChatMessage:has(div[data-testid="stChatMessageAvatarUser"]) {
        border-left-color: var(--text-muted);
    }

    div[data-testid="stChatInput"] {
        border-radius: 10px;
        border: 1px solid var(--border) !important;
        background-color: var(--surface) !important;
    }

    /* Botones generales: look de "tag" monoespaciado */
    .stButton button, .stDownloadButton button {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        border-radius: 6px !important;
        border: 1px solid var(--border) !important;
        background-color: var(--surface) !important;
        color: var(--text) !important;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Configuración de clave (para modo nube / análisis de imágenes) ----------
# Localmente viene del .env; en Streamlit Community Cloud viene de st.secrets
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    try:
        API_KEY = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        API_KEY = None

if API_KEY:
    # agent.py también la busca con os.getenv (para analyze_image, que siempre usa la nube)
    os.environ["OPENROUTER_API_KEY"] = API_KEY

# ---------- Estado de la sesión ----------
if "agent" not in st.session_state:
    st.session_state.agent = Agent(web_mode=True)

agent = st.session_state.agent

# ---------- Elegir proveedor del modelo de CHAT: nube (OpenRouter) o local (Ollama) ----------
with st.sidebar:
    st.markdown("### ✨ Mi agente")
    proveedor = st.radio(
        "Modelo de chat",
        ["☁️ Nube (OpenRouter)", "💻 Local (Ollama)"],
        index=0 if API_KEY else 1,
        key="proveedor_modelo",
    )

if proveedor.startswith("☁️"):
    if not API_KEY:
        st.error("❌ Falta OPENROUTER_API_KEY para usar el modo nube. Configúrala en tu .env, o elegí 'Local (Ollama)' en la barra lateral.")
        st.stop()
    client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
    CHAT_MODELS = [
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "google/gemma-4-31b-it:free",
    ]
else:
    # Ollama corriendo en tu propia PC, con API compatible con OpenAI
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    CHAT_MODELS = ["gemma4:e4b"]

# ---------- Barra lateral (resto de opciones) ----------
with st.sidebar:
    if proveedor.startswith("💻"):
        st.caption("🖥️ Usando `gemma4:e4b` local vía Ollama. La imagen (analyze_image) sigue usando la nube.")

    if st.button("➕ Nueva conversación", use_container_width=True):
        agent.reset()
        st.rerun()

    st.markdown("---")
    st.caption("Conversaciones")
    conversaciones = list_conversations()
    if not conversaciones:
        st.caption("_Todavía no hay conversaciones guardadas._")
    else:
        for conv in conversaciones[:15]:
            es_actual = conv["id"] == agent.conversation_id
            etiqueta = ("▶️ " if es_actual else "💬 ") + conv["title"]
            if st.button(etiqueta, use_container_width=True, key=f"conv_{conv['id']}", disabled=es_actual):
                st.session_state.agent = Agent(web_mode=True, conversation_id=conv["id"])
                st.rerun()

    st.markdown("---")

    # ---------- Descargar archivos generados en workspace/ ----------
    with st.expander("📁 Archivos en workspace"):
        archivos_encontrados = []
        for root, _, files in os.walk(SANDBOX_DIR):
            for nombre in files:
                ruta_completa = os.path.join(root, nombre)
                ruta_relativa = os.path.relpath(ruta_completa, SANDBOX_DIR)
                archivos_encontrados.append((ruta_relativa, ruta_completa))

        if not archivos_encontrados:
            st.caption("_Todavía no hay archivos._")
        else:
            for ruta_relativa, ruta_completa in sorted(archivos_encontrados):
                try:
                    with open(ruta_completa, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {ruta_relativa}",
                            data=f.read(),
                            file_name=os.path.basename(ruta_relativa),
                            use_container_width=True,
                            key=f"dl_{ruta_relativa}",
                        )
                except Exception:
                    pass

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
    ultimo_error = None
    for modelo in CHAT_MODELS:
        try:
            stream = client.chat.completions.create(
                model=modelo,
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
        except (RateLimitError, APIConnectionError, APIError) as e:
            ultimo_error = e
            continue  # probamos el siguiente modelo de la lista
    # Si ninguno funcionó, propagamos el último error para que run_agent_loop lo muestre
    raise ultimo_error


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
                if proveedor.startswith("💻"):
                    st.warning("⚠️ No se pudo conectar con Ollama. ¿Está abierto en tu PC? (la app de Ollama tiene que estar corriendo).")
                else:
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
    st.markdown("<div class='eyebrow'>// AGENTE LOCAL — WORKSPACE ACTIVO</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='greeting-title'>{saludo_segun_hora()}</div>", unsafe_allow_html=True)

# ---------- Mostrar historial de la conversación ----------
else:
    for msg in visible_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------- Menú de acciones rápidas (➕) — ejecutan por código, no dependen del modelo ----------
if not agent.pending_tool_call:
    with st.popover("➕ Acciones rápidas"):
        if st.button("📂 Listar archivos de workspace", use_container_width=True, key="qa_list"):
            resultado = agent.list_files_in_dir(".")
            archivos = resultado.get("files", []) if isinstance(resultado, dict) else []
            texto = ("Archivos en workspace:\n" + "\n".join(f"- {a}" for a in archivos)) if archivos else "La carpeta workspace está vacía."
            agent.messages.append({"role": "user", "content": "Listame los archivos que hay en workspace."})
            agent.messages.append({"role": "assistant", "content": texto})
            agent.save_history()
            st.rerun()

        st.markdown("---")
        st.markdown("**📝 Crear un archivo**")
        qa_nombre = st.text_input("Nombre del archivo (ej: notas.txt)", key="qa_filename")
        qa_contenido = st.text_area("Contenido", key="qa_filecontent", height=100)
        if st.button("Crear archivo", use_container_width=True, key="qa_create"):
            if qa_nombre:
                resultado = agent.edit_file(qa_nombre, qa_contenido or "", auto_confirmed=True)
                agent.messages.append({"role": "user", "content": f"Creá el archivo '{qa_nombre}' en workspace con el contenido que te indiqué."})
                agent.messages.append({"role": "assistant", "content": resultado})
                agent.save_history()
                st.rerun()
            else:
                st.warning("Poné un nombre de archivo primero.")

        st.markdown("---")
        st.caption("Para **analizar una imagen** o **leer un archivo externo**, adjuntalo con el 📎 de la barra de mensaje de abajo.")

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
