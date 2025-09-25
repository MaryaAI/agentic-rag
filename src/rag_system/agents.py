import os
from crewai import Agent, LLM
from .tools import document_retrieval_tool

# Initialize the Ollama LLM for the agents - using gemma3:4b with maximum tokens
# Use environment variable for base URL to support Docker deployment
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_llm = LLM(
    model="ollama/gemma3:4b",  # Using a larger model for better context understanding
    base_url=ollama_base_url,
    temperature=1,
    timeout=300,
    verbose=True,  # Enable verbose logging for debugging
    # Maximum token configuration for gemma3:4b
    max_tokens=131072,  # Use maximum context length available
    num_ctx=131072,     # Context window size
)

# --- AGENT 1: The Specialist Retriever ---
# This agent's only job is to call the retrieval tool correctly.
document_researcher = Agent(
    role='Document Researcher',
    goal='Use the Document Retrieval Tool to find information relevant to a user\'s query from the knowledge base.',
    backstory=(
   "You are an information retrieval specialist. Your role is strictly limited to:, "
   "1) Analyze the user's query to understand intent, "
   "2) Retrieve relevant text chunks using the Document Retrieval Tool, "
   "3) Return only the raw retrieved context - no interpretation or answers. "
   "DO NOT answer questions using your general knowledge. "
   "DO NOT provide explanations, summaries, or interpretations. "
   "ONLY return the exact text chunks retrieved from the tool for the next agent to use."
),
    tools=[document_retrieval_tool],
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,  # Limit iterations to prevent infinite loops
)

# --- AGENT 2: The Specialist Synthesizer ---
# This agent's only job is to write the final answer based on the context it receives.
insight_synthesizer = Agent(
    role='Insight Synthesizer',
    goal='Create clear, professional responses that directly answer user questions based on the provided context.',
    backstory=(
   "You are an expert policy analyst who specializes in creating natural, professional responses. "
   "You receive context from a document researcher and must craft responses that feel conversational yet authoritative. "
   
   "CORE PRINCIPLES: "
   "- Answer questions directly and naturally, like a knowledgeable colleague would "
   "- Use ONLY the provided context - never add outside knowledge "
   "- Adapt your response style to match the complexity of the question "
   "- Be concise for simple questions, detailed for complex ones "
   
   "RESPONSE STYLE: "
   "- Start with the most direct answer to the question "
   "- Provide supporting details naturally, not in rigid templates "
   "- Include relevant policy references and figures seamlessly in the text "
   "- Use bullet points, numbering, or paragraphs as the content naturally requires "
   "- Avoid repetitive headers like 'DIRECT ANSWER' unless genuinely needed for clarity "
   "- Make citations feel natural: 'According to Article 95...' rather than 'SOURCE REFERENCE:' "
   "- If the question is simple, keep the answer simple "
   
   "QUALITY CHECKS: "
   "- If context is insufficient, clearly state what information is missing "
   "- Ensure accuracy by staying strictly within the provided context "
   "- Maintain professional tone while being conversational "
),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,  # Limit iterations to prevent infinite loops
    # This agent does not need tools; it only processes text.
    tools=[]
)
