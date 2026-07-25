import os
import json
import base64
import mimetypes
import glob
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CONVERSATIONS_DIR = os.path.abspath("conversations")
SANDBOX_DIR = os.path.abspath("workspace")  # Carpeta segura para leer/editar archivos
MAX_MESSAGES = 20  # Mensajes recientes a conservar en memoria (sin contar el system)
MAX_EXTERNAL_FILE_SIZE = 2 * 1024 * 1024  # 2 MB, para no tragar archivos gigantes

# Herramientas que necesitan confirmación explícita del usuario antes de ejecutarse
# (tocan cosas fuera del sandbox o modifican archivos)
TOOLS_REQUIRING_CONFIRMATION = {"edit_file", "read_external_file"}

# Modelo aparte, con capacidad de visión, solo para analizar imágenes
VISION_MODEL = "google/gemma-4-31b-it:free"

SYSTEM_PROMPT = (
    "Eres un asistente útil que habla español y eres muy conciso con tus respuestas. "
    "Tenés herramientas disponibles: list_files_in_dir y read_file (para workspace/), "
    "edit_file (crear/editar archivos en workspace/), read_external_file (leer archivos fuera de workspace, con confirmación), "
    "y analyze_image (para describir imágenes Y TAMBIÉN para transcribir/leer texto dentro de una imagen, tipo OCR). "
    "Si el usuario pide algo relacionado con una imagen (describirla, analizarla, transcribir texto, saber qué dice), usá SIEMPRE analyze_image en vez de decir que no podés."
)


def list_conversations():
    """
    Devuelve la lista de conversaciones guardadas en disco (con al menos un mensaje
    de usuario), como [{"id", "title", "updated_at"}, ...], ordenadas de más
    reciente a más antigua. Se usa para mostrar el historial en la barra lateral.
    """
    if not os.path.exists(CONVERSATIONS_DIR):
        return []

    conversations = []
    for filepath in glob.glob(os.path.join(CONVERSATIONS_DIR, "*.json")):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            user_msgs = [m for m in data if m.get("role") == "user" and m.get("content")]
            if not user_msgs:
                continue  # conversación vacía, no la mostramos
            title = user_msgs[0]["content"].strip().replace("\n", " ")[:45]
            conversations.append({
                "id": os.path.splitext(os.path.basename(filepath))[0],
                "title": title,
                "updated_at": os.path.getmtime(filepath),
            })
        except Exception:
            continue

    conversations.sort(key=lambda c: c["updated_at"], reverse=True)
    return conversations


class Agent:
    def __init__(self, web_mode=False, conversation_id=None):
        """
        web_mode=False -> comportamiento original de consola (usa input() para confirmar ediciones)
        web_mode=True  -> no usa input(); las ediciones quedan "pendientes" hasta que
                          algo externo (la app de Streamlit) las confirme o rechace.
        conversation_id=None -> se crea una conversación nueva (id basado en fecha/hora).
        conversation_id=<id> -> se carga esa conversación guardada (para poder volver a
                                 chats anteriores desde la barra lateral).
        """
        self.web_mode = web_mode
        self.setup_tools()
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

        # Cliente aparte para llamar al modelo de visión (analyze_image)
        api_key = os.getenv("OPENROUTER_API_KEY")
        self._vision_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1") if api_key else None

        self.conversation_id = conversation_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.history_file = os.path.join(CONVERSATIONS_DIR, f"{self.conversation_id}.json")

        self.messages = self.load_history() or [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # ---------- Estado para confirmación en modo web ----------
        # Cuando el modelo pide edit_file en web_mode, guardamos aquí
        # la llamada a la herramienta en espera de aprobación del usuario.
        self.pending_tool_call = None  # dict: {"id","name","arguments"} o None

    # ---------- #6 Persistencia entre sesiones ----------
    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_history(self):
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ No se pudo guardar el historial: {e}")

    # ---------- #9 Comando para reiniciar ----------
    def reset(self):
        # Empezamos una conversación NUEVA (con su propio archivo);
        # la anterior queda guardada para poder volver a verla en la barra lateral.
        self.conversation_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.history_file = os.path.join(CONVERSATIONS_DIR, f"{self.conversation_id}.json")
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.pending_tool_call = None
        print("🔄 Conversación reiniciada.")

    # ---------- #2 Memoria con límite (ventana deslizante) ----------
    def trim_history(self):
        if len(self.messages) > MAX_MESSAGES + 1:
            system_msg = self.messages[0]
            recientes = self.messages[-MAX_MESSAGES:]
            self.messages = [system_msg] + recientes

    # ---------- #4 Sandbox: solo se puede tocar la carpeta workspace/ ----------
    def _resolve_path(self, path):
        full_path = os.path.abspath(os.path.join(SANDBOX_DIR, path))
        if not full_path.startswith(SANDBOX_DIR):
            raise PermissionError("Ruta fuera de la carpeta permitida (workspace/)")
        return full_path

    def setup_tools(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_files_in_dir",
                    "description": "Lista los archivos dentro de la carpeta workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "description": "Subdirectorio dentro de workspace (opcional)"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Lee el contenido de un archivo dentro de la carpeta workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Ruta relativa dentro de workspace"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edita o crea un archivo dentro de la carpeta workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Ruta relativa dentro de workspace"},
                            "prev_text": {"type": "string", "description": "Texto a reemplazar (vacío si es archivo nuevo)"},
                            "new_text": {"type": "string", "description": "Texto nuevo"}
                        },
                        "required": ["path", "new_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_external_file",
                    "description": "Lee el contenido de un archivo de texto en CUALQUIER ruta del sistema (fuera de workspace). Requiere confirmación del usuario. Usar solo cuando el usuario pida explícitamente abrir un archivo fuera de workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Ruta absoluta o relativa al archivo en el sistema"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_image",
                    "description": "Analiza, describe, o LEE/TRANSCRIBE TEXTO (funciona como OCR) dentro de una imagen (dentro de workspace o en cualquier ruta del sistema), usando un modelo con visión. Úsala también cuando el usuario pida 'transcribir', 'leer el texto de la imagen', 'qué dice esta captura', etc. — para eso, pasa en 'question' algo como 'Transcribí todo el texto que aparece en la imagen'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Ruta a la imagen (jpg, png, etc.)"},
                            "question": {"type": "string", "description": "Qué preguntar sobre la imagen (opcional, por defecto 'Describe esta imagen en detalle')"}
                        },
                        "required": ["path"]
                    }
                }
            }
        ]

    # ---------- Herramientas ----------
    def list_files_in_dir(self, directory="."):
        print("  ⚙️ Herramienta llamada: list_files_in_dir")
        try:
            full_path = self._resolve_path(directory)
            return {"files": os.listdir(full_path)}
        except Exception as e:
            return {"error": str(e)}

    def read_file(self, path):
        print("  ⚙️ Herramienta llamada: read_file")
        try:
            full_path = self._resolve_path(path)
            with open(full_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error al leer el archivo {path}: {e}"

    def edit_file(self, path, new_text, prev_text="", auto_confirmed=False):
        """
        auto_confirmed=True se usa en modo web: significa que el usuario
        ya aprobó la edición mediante los botones de la interfaz, así que
        no hay que volver a preguntar.
        """
        print("  ⚙️ Herramienta llamada: edit_file")
        try:
            full_path = self._resolve_path(path)

            # ---------- #4 Confirmación antes de modificar archivos ----------
            if not self.web_mode and not auto_confirmed:
                confirm = input(f"    ⚠️  El agente quiere modificar '{path}'. ¿Permitir? (s/n): ").strip().lower()
                if confirm != "s":
                    return "El usuario no autorizó esta edición."

            existed = os.path.exists(full_path)
            if existed and prev_text:
                content = self.read_file(path)
                if prev_text not in content:
                    return f"Texto '{prev_text}' no encontrado en el archivo"
                content = content.replace(prev_text, new_text)
            else:
                dir_name = os.path.dirname(full_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                content = new_text

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            action = "editado" if existed and prev_text else "creado"
            return f"Archivo {path} {action} exitosamente"
        except Exception as e:
            return f"Error al crear o editar el archivo {path}: {e}"

    def read_external_file(self, path, auto_confirmed=False):
        """
        Lee un archivo de texto en cualquier ruta del sistema (fuera de workspace).
        Requiere confirmación, igual que edit_file.
        """
        print("  ⚙️ Herramienta llamada: read_external_file")
        try:
            if not self.web_mode and not auto_confirmed:
                confirm = input(f"    ⚠️  El agente quiere LEER el archivo externo '{path}'. ¿Permitir? (s/n): ").strip().lower()
                if confirm != "s":
                    return "El usuario no autorizó esta lectura."

            full_path = os.path.abspath(path)
            if not os.path.exists(full_path):
                return f"El archivo {path} no existe."
            if os.path.getsize(full_path) > MAX_EXTERNAL_FILE_SIZE:
                return f"El archivo {path} es demasiado grande para leerlo (límite {MAX_EXTERNAL_FILE_SIZE // 1024 // 1024} MB)."

            with open(full_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            return f"Error al leer el archivo externo {path}: {e}"

    def analyze_image(self, path, question=None):
        """
        Codifica la imagen en base64 y se la manda a un modelo con visión
        (distinto del modelo de chat principal) para que la describa/analice.
        """
        print("  ⚙️ Herramienta llamada: analyze_image")
        if not self._vision_client:
            return "No se pudo analizar la imagen: falta OPENROUTER_API_KEY."

        try:
            # La imagen puede estar dentro de workspace o en cualquier ruta del sistema
            if os.path.exists(os.path.join(SANDBOX_DIR, path)):
                full_path = os.path.join(SANDBOX_DIR, path)
            else:
                full_path = os.path.abspath(path)

            if not os.path.exists(full_path):
                return f"No se encontró la imagen en {path}."

            mime_type, _ = mimetypes.guess_type(full_path)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/jpeg"

            with open(full_path, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode("utf-8")

            prompt = question or "Describe esta imagen en detalle."

            response = self._vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error al analizar la imagen {path}: {e}"

    def _run_tool(self, fn_name, args, auto_confirmed=False):
        if fn_name == "list_files_in_dir":
            return self.list_files_in_dir(**args)
        elif fn_name == "read_file":
            return self.read_file(**args)
        elif fn_name == "edit_file":
            return self.edit_file(**args, auto_confirmed=auto_confirmed)
        elif fn_name == "read_external_file":
            return self.read_external_file(**args, auto_confirmed=auto_confirmed)
        elif fn_name == "analyze_image":
            return self.analyze_image(**args)
        else:
            return f"Herramienta desconocida: {fn_name}"

    # ---------- Modo consola (comportamiento original) ----------
    def process_message(self, content, tool_calls):
        assistant_message = {"role": "assistant", "content": content or None}

        if tool_calls:
            assistant_message["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls.values()
            ]

        self.messages.append(assistant_message)

        if tool_calls:
            for tc in tool_calls.values():
                fn_name = tc["name"]
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}

                print(f"  - El modelo considera llamar a la herramienta {fn_name}")
                print(f"  - Argumentos: {args}")

                result = self._run_tool(fn_name, args)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps({"result": result}, ensure_ascii=False)
                })

            self.trim_history()
            return True

        self.trim_history()
        return False

    # ---------- Modo web (Streamlit) ----------
    def process_message_web(self, content, tool_calls):
        """
        Igual que process_message, pero si aparece un edit_file sin confirmar
        se detiene y deja la llamada guardada en self.pending_tool_call.

        Devuelve un string con el estado:
          "pending"  -> hay una edición esperando confirmación del usuario
          "continue" -> se ejecutaron herramientas, hay que volver a llamar al modelo
          "done"     -> no hubo tool_calls, la respuesta ya está completa
        """
        assistant_message = {"role": "assistant", "content": content or None}

        if tool_calls:
            assistant_message["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls.values()
            ]

        self.messages.append(assistant_message)

        if not tool_calls:
            self.trim_history()
            return "done"

        for tc in tool_calls.values():
            fn_name = tc["name"]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}

            if fn_name in TOOLS_REQUIRING_CONFIRMATION:
                # Pausamos aquí: guardamos la llamada y esperamos confirmación
                self.pending_tool_call = {"id": tc["id"], "name": fn_name, "args": args}
                return "pending"

            result = self._run_tool(fn_name, args)
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps({"result": result}, ensure_ascii=False)
            })

        self.trim_history()
        return "continue"

    def resolve_pending_edit(self, approved):
        """
        Se llama desde la app de Streamlit cuando el usuario aprueba
        o rechaza la acción pendiente (botones "Permitir" / "Rechazar"),
        sea edit_file o read_external_file.
        """
        if not self.pending_tool_call:
            return

        tc = self.pending_tool_call
        args = tc["args"]

        if approved:
            result = self._run_tool(tc["name"], args, auto_confirmed=True)
        else:
            result = "El usuario no autorizó esta acción."

        self.messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps({"result": result}, ensure_ascii=False)
        })

        self.pending_tool_call = None
        self.trim_history()
