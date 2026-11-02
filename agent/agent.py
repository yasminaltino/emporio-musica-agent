import random
import time

from dotenv import load_dotenv
from google.api_core.exceptions import DeadlineExceeded, ServiceUnavailable
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from requests.exceptions import ConnectionError, ReadTimeout

from tools.categories import search_categories
from tools.customers import search_customers
from tools.order_items import search_order_items
from tools.orders import search_orders
from tools.policies import search_company_policies
from tools.products import search_products
from tools.promotions import search_promotions

load_dotenv()

SYSTEM_PROMPT = """
Você é o assistente virtual da Empório da Música, uma loja de
instrumentos musicais localizada em Campo Grande, MS.

Seu objetivo é ajudar os clientes com dúvidas sobre produtos,
categorias, pedidos, clientes, promoções e políticas da loja.

REGRAS:

1. Utilize as ferramentas disponíveis para consultar informações
   dos CSVs e do documento de políticas. Não invente preços, estoque,
   produtos, pedidos, descontos ou regras da loja.

2. Para perguntas sobre produtos, consulte a ferramenta de produtos.

3. Para perguntas sobre pedidos:
    - Confirme que o pedido pertence ao cliente mencionado
    (customer_id) antes de consultar ou fornecer seus dados.
    - Consulte os itens do pedido quando necessário e utilize a
    ferramenta de produtos para obter os nomes correspondentes
    aos IDs.
    - Ao consultar produtos comprados por um cliente, considere
    somente pedidos concluídos ou entregues.
    Não considere pedidos cancelados como compras realizadas.

4. Para perguntas sobre promoções, considere somente as
   promoções ativas. Nunca informe uma promoção inativa como
   se estivesse disponível.

5. REGRAS PARA CONSULTAS ÀS POLÍTICAS:

    - Utilize os trechos retornados por search_company_policies como fonte
    para responder sobre as políticas da loja.
    - Não invente prazos, valores, condições ou exceções que não estejam
    explicitamente presentes nos documentos.
    - Se a informação não estiver disponível ou estiver incompleta,
    informe que a documentação não permite confirmar a resposta.
    - Não generalize uma regra além das condições descritas no documento.
    - Sempre que possível, informe a página de onde a informação foi obtida.

6. Combine ferramentas quando necessário para responder
   completamente à pergunta.

7. Se não encontrar a informação nos dados disponíveis,
   informe isso claramente. Não invente uma resposta.

8. Responda em português brasileiro, com linguagem natural,
   educada e objetiva.

9. Não exponha dados pessoais de clientes que não sejam
   necessários para responder à solicitação.

10. Evite chamadas repetidas às mesmas ferramentas com parâmetros
    equivalentes. Se uma consulta retornar resultados suficientes,
    utilize esses resultados em vez de repetir a pesquisa.

11. Sempre que possível, combine os filtros em uma única chamada.

12. Confie estritamente no primeiro resultado retornado pela ferramenta
    de busca de produtos. Não faça buscas redundantes ou parciais
    se a ferramenta já retornou dados para um filtro específico.
"""


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    timeout=60,
    max_retries=2,
    thinking_level="minimal",  # Velocidade instantânea de resposta
)

tools = [
    search_products,
    search_categories,
    search_orders,
    search_customers,
    search_order_items,
    search_promotions,
    search_company_policies,
]

agent = create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)


def ask_agent(question: str) -> str:
    start = time.time()
    max_attempts = 3

    retryable_errors = (
        DeadlineExceeded,
        ServiceUnavailable,
        ReadTimeout,
        ConnectionError,
        TimeoutError,
    )

    for attempt in range(max_attempts):
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            break

        except retryable_errors as e:
            if attempt == max_attempts - 1:
                return (
                    "Estou com instabilidade temporária ao consultar as informações. "
                    "Por favor, tente novamente em alguns instantes."
                )

            delay = (2**attempt) + random.uniform(0, 0.5)
            time.sleep(delay)

    for message in response["messages"]:
        # Chamadas feitas pelo modelo
        if message.type == "ai" and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\n[TOOL]: {tool_call['name']}")
                print(f"[ARGS]: {tool_call['args']}")

        # Resultados retornados pelas ferramentas
        elif message.type == "tool":
            print(f"\n[TOOL RESULT]: {message.name}")
            print(f"[OUTPUT]: {message.content}")

    print(f"\n[TEMPO TOTAL]: {time.time() - start:.2f}s")

    last_message = response["messages"][-1]
    content = last_message.content

    # Se o conteúdo vier em formato de lista (com tipos, extras e signatures), extrai apenas o texto
    if isinstance(content, list):
        text_parts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and "text" in part
        ]
        return "\n".join(text_parts)

    return str(content)


if __name__ == "__main__":
    print("Empório da Música - Assistente Virtual")
    print("Digite 'sair' para encerrar.\n")

    while True:
        question = input("Você: ")

        if question.lower().strip() == "sair":
            break

        try:
            answer = ask_agent(question)
            print(f"\nAssistente: {answer}\n")

        except Exception as error:
            print(f"\nErro ao consultar o agente: {error}\n")
