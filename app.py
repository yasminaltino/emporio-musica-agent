import streamlit as st

from agent.agent import ask_agent

st.set_page_config(
    page_title="Empório da Música",
    page_icon="🎸",
    layout="centered",
)


# Estado da conversa
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Olá! Seja bem-vindo à Empório da Música! 🎸\n\n"
                "Sou o assistente virtual da loja. Como posso ajudar?"
            ),
        }
    ]


# Cabeçalho
st.title("🎸 Empório da Música")
st.caption("Assistente virtual · Sua música começa aqui.")

# Nova conversa
if st.button("＋ Nova conversa"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Olá! Seja bem-vindo à Empório da Música! 🎸\n\nComo posso ajudar?"
            ),
        }
    ]
    st.rerun()


st.divider()


# Exemplos de perguntas
st.markdown("**Experimente perguntar:**")

examples = [
    "Quais violões estão disponíveis até R$ 700?",
    "Quais produtos estão em promoção?",
    "Qual é o status do pedido 1?",
    "Quais são os prazos de entrega?",
]

cols = st.columns(2)

for index, question in enumerate(examples):
    if cols[index % 2].button(question, use_container_width=True):
        st.session_state.pending_question = question


# Histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Entrada de mensagem
prompt = st.chat_input("Digite sua mensagem...")


# Pergunta selecionada nos exemplos ou digitada
if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"), st.spinner("Consultando as informações..."):
        try:
            response = ask_agent(prompt)
            st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception:
            st.error(
                "Não foi possível processar sua solicitação. "
                "Tente novamente em instantes."
            )
