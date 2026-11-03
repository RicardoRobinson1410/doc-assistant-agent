from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def get_intent_classification_prompt() -> PromptTemplate:
    """
    Get the intent classification prompt template.
    Guides the LLM to classify user intent with categories, confidence scoring, and reasoning.
    """
    return PromptTemplate(
        input_variables=["user_input", "conversation_history"],
        template="""You are an expert intent classification assistant for an intelligent document processing system.

Analyze the user input and recent conversation history, then classify the user's intent into exactly ONE of these categories:
- qa: Questions asking for specific information, facts, or data points from documents that do not require mathematical operations.
- summarization: Requests to summarize, synthesize, or extract key highlights/overviews from one or more documents.
- calculation: Requests requiring numerical computation, mathematical operations, totals, differences, percentages, or questions requiring calculating with values from documents.
- unknown: Requests that cannot be determined, are off-topic, or do not fit any of the above categories.

Instructions:
1. Assign a confidence score between 0.0 and 1.0 representing how certain you are in this classification.
2. Provide a concise 1-2 sentence explanation of your reasoning.
3. Always choose one of the four categories: "qa", "summarization", "calculation", or "unknown".

## Few-Shot Examples

Example 1:
User Input: "What is the billing address in invoice INV-001?"
Intent:
intent_type: "qa"
confidence: 0.95
reasoning: "The user is querying a specific document field without requiring mathematical computation."

Example 2:
User Input: "Can you provide a summary of the Service Agreement contract CON-001?"
Intent:
intent_type: "summarization"
confidence: 0.98
reasoning: "The user explicitly asks for a summary of a contract document."

Example 3:
User Input: "Calculate the total cost if we add invoice INV-001 and claim CLM-001."
Intent:
intent_type: "calculation"
confidence: 0.99
reasoning: "The request involves mathematical addition of amounts from multiple documents."

Example 4:
User Input: "What is 15% of the total contract value in CON-001?"
Intent:
intent_type: "calculation"
confidence: 0.96
reasoning: "The user requests calculating a percentage of a document value, requiring arithmetic calculation."

Example 5:
User Input: "What's the weather like today in Seattle?"
Intent:
intent_type: "unknown"
confidence: 0.90
reasoning: "The query is unrelated to document querying, summarization, or financial/healthcare calculation."

---
User Input: {user_input}

Recent Conversation History:
{conversation_history}

Classify the user intent, confidence score, and reasoning:
"""
    )


# Q&A System Prompt
QA_SYSTEM_PROMPT = """You are a helpful document assistant specializing in answering questions about financial and healthcare documents.

Your capabilities:
- Answer specific questions about document content
- Cite sources accurately
- Provide clear, concise answers
- Use available tools to search and read documents

Guidelines:
1. Always search for relevant documents before answering
2. Cite specific document IDs when referencing information
3. If information is not found, say so clearly
4. Be precise with numbers and dates
5. Maintain professional tone
"""

# Summarization System Prompt
SUMMARIZATION_SYSTEM_PROMPT = """You are an expert document summarizer specializing in financial and healthcare documents.

Your approach:
- Extract key information and main points
- Organize summaries logically
- Highlight important numbers, dates, and parties
- Keep summaries concise but comprehensive

Guidelines:
1. First search for and read the relevant documents
2. Structure summaries with clear sections
3. Include document IDs in your summary
4. Focus on actionable information
"""

# Calculation System Prompt
CALCULATION_SYSTEM_PROMPT = """You are an accurate calculation document assistant specializing in financial and healthcare documents.

Responsibilities:
1. Determine which document or documents must be retrieved and retrieve them using the document search and document reader tools.
2. Carefully extract the numerical values, dates, percentages, and units required for the calculation.
3. Determine the exact mathematical expression to calculate based on the user's input and document data.
4. Use the calculator tool to perform the calculation.
5. You MUST use the calculator tool for ALL calculations, no matter how simple (even basic addition like 2 + 2). Never perform mental math.
6. Present the final result clearly with units, the expression evaluated, and cite all source document IDs used.

Rules:
- NEVER perform mental math or guess numerical outcomes.
- ALWAYS call the calculator tool for every arithmetic calculation.
- ALWAYS cite the document IDs where the numbers originated.
- If data needed to complete the calculation is missing from the documents, clearly explain what is missing.
"""


def get_chat_prompt_template(intent_type: str) -> ChatPromptTemplate:
    """
    Get the appropriate chat prompt template based on intent.
    Supports "qa", "summarization", and "calculation".
    """
    if intent_type == "qa":
        system_prompt = QA_SYSTEM_PROMPT
    elif intent_type == "summarization":
        system_prompt = SUMMARIZATION_SYSTEM_PROMPT
    elif intent_type == "calculation":
        system_prompt = CALCULATION_SYSTEM_PROMPT
    else:
        system_prompt = QA_SYSTEM_PROMPT  # Default fallback

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])


# Memory Summary Prompt
MEMORY_SUMMARY_PROMPT = """Summarize the following conversation history into a concise summary:

Focus on:
- Key topics discussed
- Documents referenced
- Important findings or calculations
- Any unresolved questions
"""
