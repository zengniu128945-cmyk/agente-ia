import os
import json

HISTORY_FILE = "historial.json"
SANDBOX_DIR = os.path.abspath("workspace")  # Carpeta segura para leer/editar archivos
MAX_MESSAGES = 20  # Mensajes recientes a conservar en memoria (sin contar el system)

SYSTEM_PROMPT = "Eres un asistente útil que habla español y eres muy conciso con tus respuestas"


class Agent:
    def __init__(self, web_mode=False):
        """
        web_mode=False -> comportamiento original de consola (usa input() para confirmar ediciones)
        web_mode=True  -> no usa input(); las ediciones quedan "pendientes" hasta que
                          algo externo (la app de Streamlit) las confirme o rechace.
        """
        self.web_mode = web_mode
        self.setup_tools()
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        self.messages = self.load_history() or [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # ---------- Estado para confirmación en modo web ----------
        # Cuando el modelo pide edit_file en web_mode, guardamos aquí
        # la llamada a la herramienta en espera de aprobación del usuario.
        self.pending_tool_call = None  # dict: {"id","name","arguments"} o None

    # ---------- #6 Persistencia entre sesiones ----------
    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ No se pudo guardar el historial: {e}")

    # ---------- #9 Comando para reiniciar ----------
    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.pending_tool_call = None
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
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

    def _run_tool(self, fn_name, args, auto_confirmed=False):
        if fn_name == "list_files_in_dir":
            return self.list_files_in_dir(**args)
        elif fn_name == "read_file":
            return self.read_file(**args)
        elif fn_name == "edit_file":
            return self.edit_file(**args, auto_confirmed=auto_confirmed)
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

            if fn_name == "edit_file":
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
        o rechaza la edición pendiente (botones "Permitir" / "Rechazar").
        """
        if not self.pending_tool_call:
            return

        tc = self.pending_tool_call
        args = tc["args"]

        if approved:
            result = self.edit_file(**args, auto_confirmed=True)
        else:
            result = "El usuario no autorizó esta edición."

        self.messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps({"result": result}, ensure_ascii=False)
        })

        self.pending_tool_call = None
        self.trim_history()