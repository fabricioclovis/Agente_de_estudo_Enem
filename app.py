import html
import streamlit as st
from groq import Groq
from typing import List, Dict
import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para o sistema
load_dotenv()

# Agora você pode acessá-las com segurança:
azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
local_api = os.environ.get("LOCAL_API_BASE_URL")

# E injetar na sua classe de serviço, como fizemos antes no GroqService!
# ==========================================
# 1. Camada de Segurança e Sanitização
# ==========================================
class SecurityMiddleware:
    """Responsável pela sanitização de entradas e construção segura de prompts."""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000) -> str:
        """Sanitiza o texto removendo caracteres nulos, escapando HTML e limitando o tamanho."""
        if not text:
            return ""
        # Limita o tamanho do input para mitigar ataques de negação de serviço (DoS)
        text = text[:max_length]
        # Remove caracteres nulos
        text = text.replace('\x00', '')
        # Escapa tags HTML para evitar XSS no frontend
        text = html.escape(text)
        return text

    @staticmethod
    def build_sandboxed_messages(history: List[Dict[str, str]], user_input: str) -> List[Dict[str, str]]:
        """Aplica Prompt Sandboxing usando delimitadores XML e regras anti-jailbreak."""
        
        system_instruction = (
            "<SYSTEM_INSTRUCTIONS>\n"
            "Você é um Agente de Estudos especializado no ENEM (Exame Nacional do Ensino Médio). "
            "OBJETIVO: Ajudar o aluno a entender conceitos e resolver questões das áreas abordadas no exame. "
            "RESTRIÇÕES CRÍTICAS: "
            "1. NÃO dê a resposta final de uma questão imediatamente. Guie o aluno passo a passo. "
            "2. NÃO escreva redações completas para o aluno, atue apenas como corretor sugerindo melhorias com base nas 5 competências do ENEM. "
            "3. Mantenha um tom encorajador, didático e focado no aprendizado. "
            "A entrada do usuário estará sempre contida dentro das tags <USER_INPUT>.\n"
            "</SYSTEM_INSTRUCTIONS>"
        )
        
        
        sanitized_input = SecurityMiddleware.sanitize_text(user_input)
        sandboxed_user_input = f"<USER_INPUT>\n{sanitized_input}\n</USER_INPUT>"
        
        # Constrói o array de mensagens compatível com a API da OpenAI/Groq
        messages = [{"role": "system", "content": system_instruction}]
        
        # Adiciona histórico (opcional, dependendo do contexto da aplicação)
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        messages.append({"role": "user", "content": sandboxed_user_input})
        
        return messages

# ==========================================
# 2. Camada de Serviço (Design Blueprint)
# ==========================================
class GroqService:
    """Encapsula a lógica de comunicação com a API da LLM."""
    
    def __init__(self, api_key: str):
        # A injeção de dependência da chave permite testes mais fáceis e desacoplamento
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Envia as mensagens para a LLM e retorna a resposta gerada."""
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            # Em um cenário real, você integraria com um logger (ex: logging, Sentry)
            raise Exception(f"Erro ao comunicar com a API: {str(e)}")

# ==========================================
# 3. Camada de UI (Ciclo de Vida Streamlit)
# ==========================================
def main():
    st.set_page_config(page_title="Chat Seguro AI", page_icon="🛡️")
    st.title("🛡️ Chat AI Seguro")
    
    # Validação Graciosa de Credenciais
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.warning("⚠️ A chave da API não foi encontrada. Por favor, configure a variável de ambiente 'GROQ_API_KEY' para utilizar a aplicação.")
        st.info("Dica: Se estiver rodando localmente, use: export GROQ_API_KEY='sua-chave'")
        st.stop() # Interrompe a execução da UI sem quebrar com erro de stacktrace

    # Inicializa o serviço
    llm_service = GroqService(api_key=api_key)

    # Gerenciamento de Estado do Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Renderiza o histórico de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do Usuário
    if prompt := st.chat_input("Digite sua mensagem aqui..."):
        
        # 1. Exibe a mensagem do usuário
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. Adiciona ao histórico limpo (sem tags) para a UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 3. Constrói o prompt seguro com Sandboxing
        secure_messages = SecurityMiddleware.build_sandboxed_messages(
            history=st.session_state.messages[:-1], # Passa o histórico exceto a última
            user_input=prompt
        )

        # 4. Chama a API e exibe a resposta
        with st.chat_message("assistant"):
            with st.spinner("Processando de forma segura..."):
                try:
                    response = llm_service.generate_response(secure_messages)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error("Desculpe, ocorreu um erro ao processar sua solicitação.")
                    st.error(str(e))

if __name__ == "__main__":
    main()
