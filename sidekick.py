from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
from sidekick_tools import playwright_tools, other_tools
import uuid
import asyncio
from datetime import datetime
import os

load_dotenv(override=True)


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    clarification_questions: List[str]
    clarifications_collected: int
    current_specialist: Optional[str]
    specialist_history: List[str]


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


class ClarificationOutput(BaseModel):
    questions: List[str] = Field(description="List of up to 3 clarifying questions")
    needs_clarification: bool = Field(description="Whether clarification is needed")


class Sidekick:
    def __init__(self, user_id: str = "default"):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = None
        self.llm_with_tools = None
        self.graph = None
        self.user_id = user_id
        self.sidekick_id = str(uuid.uuid4())
        self.current_thread_id = str(uuid.uuid4())
        self.clarifications_done = False
        self.memory = SqliteSaver.from_conn_string(f"sandbox/memory_{user_id}.db")
        self.browser = None
        self.playwright = None

    async def setup(self):
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()
        worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        await self.build_graph()

    def worker(self, state: State) -> Dict[str, Any]:
        system_message = f"""You are a helpful assistant that can use tools to complete tasks.
    You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
    You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
    You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
    The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    IMPORTANT: Whenever you create a Markdown (.md) file using the write_file tool, you MUST automatically convert it to PDF format:
    1. First use write_file to create the .md file (e.g., 'report.md')
    2. Then IMMEDIATELY use markdown_to_pdf to convert it to PDF (e.g., 'report.md' -> 'report.pdf')
    3. Both files should be in the 'sandbox' directory
    4. Always create the PDF version - do not skip this step!

    SQL DATABASE TOOL:
    You have access to a SQLite database via sql_query tool. Use it to:
    - Store user preferences (CREATE TABLE preferences ...)
    - Remember information across conversations (INSERT, UPDATE)
    - Query historical data (SELECT)
    Database location: sandbox/data.db (auto-created if needed)

    This is the success criteria:
    {state["success_criteria"]}
    You should reply either with a question for the user about this assignment, or with your final response.
    If you have a question for the user, you need to reply by clearly stating your question. An example might be:

    Question: please clarify whether you want a summary or a detailed answer

    If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
    """

        if state.get("feedback_on_work"):
            system_message += f"""
    Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
    Here is the feedback on why this was rejected:
    {state["feedback_on_work"]}
    With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

        # Add in the system message

        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True

        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

        # Invoke the LLM with tools
        response = self.worker_llm_with_tools.invoke(messages)

        # Return updated state
        return {
            "messages": [response],
        }

    def worker_router(self, state: State) -> str:
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"

    def format_conversation(self, messages: List[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    def evaluator(self, state: State) -> State:
        last_response = state["messages"][-1].content

        system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
    Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
    and whether more input is needed from the user."""

        user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

    The entire conversation with the assistant, with the user's original request and all replies, is:
    {self.format_conversation(state["messages"])}

    The success criteria for this assignment is:
    {state["success_criteria"]}

    And the final response from the Assistant that you are evaluating is:
    {last_response}

    Respond with your feedback, and decide if the success criteria is met by this response.
    Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

    The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.
    Overall you should give the Assistant the benefit of the doubt if they say they've done something. But you should reject if you feel that more work should go into this.

    """
        if state["feedback_on_work"]:
            user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        new_state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Evaluator Feedback on this answer: {eval_result.feedback}",
                }
            ],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }
        return new_state

    def clarification_agent(self, state: State) -> Dict[str, Any]:
        """Ask up to 3 claryfying questions before proceeding"""
        # Skip if already asked clarifications in this conversation
        if state.get("clarifications_collected", 0) > 0:
            print(f"[DEBUG] Skipping clarification - already collected: {state.get('clarifications_collected')}")
            return {"clarifications_collected": state["clarifications_collected"]}

        clarification_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(ClarificationOutput)

        system_message = """You are a clarification specialist. Based on the user's request generate
        up to 3 important clarifying questions that would help complete the task better.
        Only ask questions if they would significantly improve the outcome.

        IMPORTANT: Respond in the SAME LANGUAGE as the user's request. If the user writes in Polish, respond in Polish.
        If in English, respond in English. Match the user's language exactly."""

        # Get user request content (handle both string and HumanMessage)
        first_msg = state['messages'][0]
        user_request = first_msg.content if hasattr(first_msg, 'content') else first_msg

        user_message = f"User request: {user_request}\nSuccess criteria: {state['success_criteria']}"

        result = clarification_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ])

        if result.needs_clarification and result.questions:
            # Detect user's language and create appropriate intro message

            # Ask LLM to create intro in user's language
            intro_llm = ChatOpenAI(model="gpt-4o-mini")
            intro_prompt = f"""Generate a brief introduction message in the SAME LANGUAGE as this request: "{user_request}"

            The message should say something like "Before I proceed, I have these clarifying questions:" but in the request's language.
            Reply ONLY with the intro message, nothing else."""

            intro_result = intro_llm.invoke([HumanMessage(content=intro_prompt)])
            intro_text = intro_result.content.strip()

            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.questions)])
            return {
                "messages": [AIMessage(content=f"{intro_text}\n\n{questions_text}")],
                "user_input_needed": True,
                "clarification_questions": result.questions,
                "clarifications_collected": 1
            }

        return {"clarifications_collected": 1}


    def coordinator_agent(self, state: State) -> Dict[str, Any]:
        """Decides which specialist should hande task"""

        #get user request
        first_msg = state['messages'][0]
        user_request = first_msg.content if hasattr(first_msg, 'content') else first_msg

        #LLm choose specialist
        coordinator_llm = ChatOpenAI(model="gpt-4o-mini")

        system_message = """You are a task coordinator. Analyze the user's request and decide
        which specialist should handle it.

    Available specialists:
    - "research": For web searches, finding information, browsing websites, Wikipedia lookups
    - "coding": For writing/executing Python code, file operations, creating documents, markdown/PDF generation
    - "communication": For sending notifications, emails, or other communication tasks
    - "general": For simple questions, conversations, or tasks that don't fit other categories

    Respond ONLY ONE WORD: research, coding, communication or general"""

        user_message = f"User request: {user_request}\n\nWhich specialist should handle this?"

        result = coordinator_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ])

        specialist = result.content.strip().lower()

        #Response validation
        if specialist not in ["research", "coding", "communication", "general"]:
            specialist = "general"
        print(f"[COORDINATOR] Routing to: {specialist}")

        #Update specialist history
        history = state.get("specialist_history", [])
        history.append(specialist)

        return {
            "current_specialist": specialist,
            "specialist_history": history
        }

    def research_agent(self, state: State) -> Dict[str, Any]:
        """Specialized agent for research tasks - web search, browsing, Wikipedia"""

        system_message = f"""You are a research specialist. Your job is to gather infromation using:
    - Web search (search tool)
    - Wikipedia (wikipedia_run_tool)
    - Web browsing (navigate_browser, get_elements tools)
    
    Focus ONLY on research and information gathering. Be through and cite sources.
    The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%s")}
    
    Success criteria: {state["success_criteria"]}
    """

        if state.get("feedback_on_work"):
            system_message += f"""
    Previously your research was incomplete. Feedback:
    {state["feedback_on_work"]}
    Please improve your research based on this feedback."""

        research_tool_names = ["search", "wikipedia_run", "navigate_browser", "get_elements", "click"]
        research_tools = [t for t in self.tools if t.name in research_tool_names]

        research_llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(research_tools)

        found_system = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system = True

        if not found_system:
            messages = [SystemMessage(content=system_message)] + messages

        response = research_llm.invoke(messages)

        return {"messages": [response]}

    def coding_agent(self, state: State) -> Dict[str, Any]:
        """Specialized agent for coding tasks - Python execution, file operations, document creation"""

        system_message = f"""You are a coding specialist. Your job is to:
    - Execute Python code (python_repl tool)
    - Manipulate files (write_file, read_file, list_directory, copy_file, move_file, file_delete)
    - Create documents (write markdown files, then convert to PDF with markdown_to_pdf)

    IMPORTANT: When creating markdown documents, ALWAYS convert them to PDF:
    1. Use write_file to create the .md file (e.g., 'report.md')
    2. IMMEDIATELY use markdown_to_pdf to convert (e.g., 'sandbox/report.md' -> 'sandbox/report.pdf')
    3. Both files should be in 'sandbox' directory

    The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    Success criteria: {state["success_criteria"]}
    """

        if state.get("feedback_on_work"):
            system_message += f"""
    Previously your code/work was incomplete. Feedback:
    {state["feedback_on_work"]}
    Please fix the issues based on this feedback."""

        # Filter tools to coding-only
        coding_tool_names = [
            "python_repl", "write_file", "read_file", "list_directory",
            "copy_file", "move_file", "file_delete", "markdown_to_pdf"
        ]
        coding_tools = [t for t in self.tools if t.name in coding_tool_names]

        # Bind LLM with coding tools only
        coding_llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(coding_tools)

        # Prepare messages
        found_system = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system = True

        if not found_system:
            messages = [SystemMessage(content=system_message)] + messages

        # Invoke LLM
        response = coding_llm.invoke(messages)

        return {"messages": [response]}


    def communication_agent(self, state: State) -> Dict[str, Any]:
        """Specialized agent for communication tasks - notifications, emails"""

        system_message = f"""You are a communication specialist. Your job is to:
    - Send push notifications (send_push_notification tool)
    - Handle user communication tasks

    Be concise and clear in your communications.
    The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    Success criteria: {state["success_criteria"]}
    """

        if state.get("feedback_on_work"):
            system_message += f"""
    Previously your communication was unclear. Feedback:
    {state["feedback_on_work"]}
    Please improve based on this feedback."""

        comm_tool_names = ["send_push_notification"]
        comm_tools = [t for t in self.tools if t.name in comm_tool_names]
        comm_llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(comm_tools)

        found_system = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system = True

        if not found_system:
            messages = [SystemMessage(content=system_message)] + messages

        response = comm_llm.invoke(messages)

        return {"messages": [response]}

    def specialist_router(self, state: State) -> str:
        """Route to appropriate specialist or back to tools"""
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"


    def route_based_on_evaluation(self, state: State) -> str:
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"

    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("coordinator", self.coordinator_agent)
        graph_builder.add_node("research_agent", self.research_agent)
        graph_builder.add_node("coding_agent", self.coding_agent)
        graph_builder.add_node("communication_agent", self.communication_agent)
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)
        graph_builder.add_node("clarification", self.clarification_agent)

        def route_to_specialist(state: State) -> str:
            specialist = state.get("current_specialist", "general")
            if specialist == "research":
                return "research_agent"
            elif specialist == "coding":
                return "coding_agent"
            elif specialist == "communication":
                return "communication_agent"
            else:
                return "worker"

        # Add edges
        graph_builder.add_edge(START, "clarification")

        # Clarification routing
        graph_builder.add_conditional_edges(
            "clarification",
            lambda state: "END" if state.get("user_input_needed") else "coordinator",
            {"coordinator": "coordinator", "END": END}
        )

        # Coordinator routes to specialists
        graph_builder.add_conditional_edges(
            "coordinator",
            route_to_specialist,
            {
                "research_agent": "research_agent",
                "coding_agent": "coding_agent",
                "communication_agent": "communication_agent",
                "worker": "worker"
            }
        )

        # Each specialist routes to tools or evaluator
        for specialist_node in ["research_agent", "coding_agent", "communication_agent", "worker"]:
            graph_builder.add_conditional_edges(
                specialist_node,
                self.specialist_router,
                {"tools": "tools", "evaluator": "evaluator"}
            )

        # Tools always go back to coordinator (might need different specialist)
        graph_builder.add_edge("tools", "coordinator")

        # Evaluator decides: continue (back to coordinator) or end
        graph_builder.add_conditional_edges(
            "evaluator",
            self.route_based_on_evaluation,
            {"worker": "coordinator", "END": END}  # Changed "worker" to "coordinator"
        )

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

        

    async def run_superstep(self, message, success_criteria, history):
        # Use different thread_id for each new conversation to reset clarifications
        # If this is first message (empty history), create new thread
        if not history:
            self.current_thread_id = str(uuid.uuid4())
            self.clarifications_done = False

        config = {"configurable": {"thread_id": self.current_thread_id}}

        # Create state with user message as HumanMessage for the graph
        # Only ask clarifications on first message
        state = {
            "messages": [HumanMessage(content=message)],
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "clarification_questions": [],
            "clarifications_collected": 1 if self.clarifications_done else 0,
            "current_specialist": None,
            "specialist_history": [],
        }
        result = await self.graph.ainvoke(state, config=config)

        # Mark clarifications as done after first interaction
        if result.get("user_input_needed"):
            self.clarifications_done = True

        # For Gradio UI history:
        # 1. Add user message (for display in chatbot)
        # 2. Add assistant's main reply
        # 3. Add evaluator feedback (if present)

        user_msg = {"role": "user", "content": message}

        # Get assistant messages from result
        assistant_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]

        if len(assistant_messages) >= 2:
            # Normal case: reply + feedback
            reply = {"role": "assistant", "content": assistant_messages[-2].content}
            feedback = {"role": "assistant", "content": assistant_messages[-1].content}
            return history + [user_msg, reply, feedback]
        elif len(assistant_messages) == 1:
            # Only one message (e.g., clarification questions)
            reply = {"role": "assistant", "content": assistant_messages[-1].content}
            return history + [user_msg, reply]
        else:
            # Fallback
            return history + [user_msg]

    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())
